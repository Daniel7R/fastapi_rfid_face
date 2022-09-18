import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from typing import Optional

import cv2
import face_recognition

import psycopg2 as conn
import numpy as np

from dotenv import load_dotenv
import os
from datetime import datetime

# Importo las funciones creadas en el otro mmodulo

load_dotenv()
app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

date_format_str = '%Y-%m-%d %H:%M:%S.%f%z'

known_path = "./Images/Known Faces/"
unknown_path = "./Images/Unknown Faces/"


date_format_str = '%Y-%m-%d %H:%M:%S.%f%z'

db = []


def get_data():
    global db
    con = conn.connect(host=os.environ["HOST"], database=os.environ["DB"],
                       user=os.environ["USER"], password=os.environ["PASSWORD"], port=os.environ["PORT"])
    # Driver para la conexión a la base de datos
    cursor = con.cursor()

    sql = "SELECT * FROM usuarios;"
    cursor.execute(sql)

    result = cursor.fetchall()

    for i in result:
        l = []
        # Add all the db fields
        l.append(i[0])
        l.append(i[1])
        l.append(i[2])
        l.append(i[3])
        l.append(i[4])
        l.append(i[5])
        l.append(i[6])
        l.append(i[7])
        l.append(i[8])
        l.append(i[9])
        # This is for the face field
        string = i[10][1:-3]
        nums = []

        for j in string.split():
            nums.append(float(j.strip()))
        l.append(nums)
        db.append(l)

    # Se cierra la conexión a la base de datos
    cursor.close()
    con.close()


def set_in_time(id):
    con = conn.connect(host=os.environ["HOST"], database=os.environ["DB"],
                       user=os.environ["USER"], password=os.environ["PASSWORD"],
                       port=os.environ["PORT"])

    cursor = con.cursor()
    sql = "UPDATE usuarios SET in_time=%s, out_time = '2000/01/01 00:00:00' WHERE rfid=%s"

    now = datetime.now()
    current_time = now.strftime(date_format_str)

    values = (current_time, id)

    cursor.execute(sql, values)
    con.commit()
    cursor.close()
    con.close()


# Sacar la diferenci entre entrada y salida para agregar al acumulador

def setHoursAccumulatorBetweenIn2Out(in_date, out_date) -> float:

    start = datetime.strptime(in_date, date_format_str)
    end = datetime.strptime(out_date, date_format_str)

    diff = end - start
    diff_in_hours = diff.total_seconds() / 3600

    return diff_in_hours


def set_out_time(id):
    con = conn.connect(host=os.environ["HOST"], database=os.environ["DB"],
                       user=os.environ["USER"], password=os.environ["PASSWORD"],
                       port=os.environ["PORT"])

    cursor = con.cursor()
    sql = "SELECT in_time from usuarios WHERE rfid='%s'" % (id)
    cursor.execute(sql)

    result = cursor.fetchall()

    in_time = result[0][0]

    # Setting the out time
    now = datetime.now()
    current_time = now.strftime(date_format_str)
    values = (current_time, id)
    sql1 = "UPDATE usuarios SET out_time=%s WHERE rfid=%s"
    cursor.execute(sql1, values)
    # con.commit()

    # Setting accumulator time(worked hours)
    queryOutTime = "SELECT out_time FROM usuarios WHERE rfid='%s'" % (
        id)
    cursor.execute(queryOutTime)
    result = cursor.fetchall()
    out_time = result[0][0]

    queryAcc = "SELECT accumulator FROM usuarios WHERE rfid='%s'" % (id)
    cursor.execute(queryAcc)

    result = cursor.fetchall()

    total = result[0][0]

    diff_horas = setHoursAccumulatorBetweenIn2Out(str(in_time), str(out_time))
    print("Diff", diff_horas)
    total = float(total) + float(diff_horas)

    total = round(total, 3)
    print("Total: ", total)
    # Insert accumulator updated
    sqlUA = "UPDATE usuarios SET accumulator=%s WHERE rfid='%s'" % (total, id)
    cursor.execute(sqlUA)

    con.commit()
    cursor.close()
    con.close()


@app.get("/")
def home():
    return "Working!"


@app.get(path="/register", summary="Register a user", tags=["Register"])
def register(
    id: Optional[str], name: Optional[str],
    edad: Optional[int], genero: Optional[str],
    estrato: Optional[int], departamento: Optional[str],
    rfid: Optional[str]
):
    con = conn.connect(host=os.environ["HOST"], database=os.environ["DB"],
                       user=os.environ["USER"], password=os.environ["PASSWORD"],
                       port=os.environ["PORT"])

    cursor = con.cursor()
    sql = "INSERT INTO usuarios VALUES(%s, %s,%s,%s,%s,%s,%s,%s,%s,%s,%s);"

    # Obtengo momento actual
    now = datetime.now()
    current_time = now.strftime(date_format_str)

    in_time = current_time
    out_time = "2000/01/01 00:00:00"
    accumulator = 0
    accumulator = float(accumulator)
    video_capture = cv2.VideoCapture(0)

    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.24, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(
        rgb_small_frame, face_locations)

    # To save in the folder of known persons (registered)
    dir = known_path + name

    if (not os.path.isdir(dir)):
        os.mkdir(dir)

    rand_no = np.random.random_sample()
    # Save and destroy camera instance
    cv2.imwrite(dir + "/"+str(rand_no) + ".jpg", frame)
    video_capture.release()
    cv2.destroyAllWindows()

    encoding = ""

    for i in face_encodings:
        encoding += str(i) + ","

    aux = [id, name, edad, genero, estrato, departamento, rfid,
           in_time, out_time, accumulator, encoding]
    value = tuple(aux)
    cursor.execute(sql, value)
    con.commit()
    cursor.close()
    con.close()

    return JSONResponse(content={"status": "Done"}, media_type="application/json")


@app.get(path="/login-with-rfid", summary="Login with RfId", tags=["Login"])
def loginRfId(rfid: Optional[str] = None):
    get_data()
    global db

    if db == []:
        msg = "You are unknown, first register yourself"
        status = "Error"
    else:
        known_rfids = [i[6] for i in db]
        known_names = [i[1] for i in db]
        # rfid = request.args.get("rfid")
        set_in_time(rfid)

        idx = -1
        for i in range(len(known_rfids)):

            if known_rfids[i] == rfid:
                idx = i

        if idx != -1:
            msg = known_names[idx]
            status = "ok"
        else:
            msg = "Rfid unknown"
            status = "error"

    return JSONResponse(content={"data": msg, "status": status}, media_type="application/json")


@app.get(path="/login-with-face", summary="Login user with face", tags=["Login"])
def loginFace(a):
    get_data()
    global db

    if db == []:
        msg = "You are unknown, first register yourself"
    else:
        # Obtengo los datos de la db
        known_face_encodings = [i[-1] for i in db]
        known_rfids = [i[6] for i in db]
        known_face_names = [i[1] for i in db]
        face_locations = []
        face_encodings = []
        face_names = []

        img_capture = cv2.VideoCapture(0)
        ret, frame = img_capture.read()

        # Se reajusta la foto en una de menor tamaño para que sea mucho mas facil procesarla
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Se llevan todas las imagenes a un solo canal de color, red
        rgb_small_frame = small_frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)

        if face_encodings == []:
            msg = "You are unknown, first regiter yourself"
            status = "unknown"
        else:

            for face_encoding in face_encodings:
                # Comparo las caras
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding, tolerance=0.75)
                name = "Unknown"
                # Hallo distancia de las caras, se usa ecuación distancia-punto
                face_distances = face_recognition.face_distance(
                    known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                print(face_distances)

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    set_in_time(known_rfids[best_match_index])

                if name != "Unknown":
                    msg = name
                    status = "ok"
                else:
                    msg = "You are unknown, first register yourself"
                    status = "You are unknown"
                face_names.append(name)

            rand_no = np.random.random_sample()
            cv2.imwrite(unknown_path+str(rand_no)+".jpg", frame)

    return JSONResponse(content={"data": msg, "status": status}, media_type="application/json")

# To logout

# With face id


@app.get(path="/logout-with-face", summary="Logout with face", tags=["Logout"])
def logoutFace():
    get_data()
    global db

    # Obtengo los datos de la db
    known_face_encodings = [i[-1] for i in db]
    known_rfids = [i[6] for i in db]
    known_face_names = [i[1] for i in db]
    face_locations = []
    face_encodings = []
    face_names = []

    img_capture = cv2.VideoCapture(0)
    ret, frame = img_capture.read()

    # Se reajusta la foto en una de menor tamaño para que sea mucho mas facil procesarla
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Se llevan todas las imagenes a un solo canal de color, red
    rgb_small_frame = small_frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(
        rgb_small_frame, face_locations)

    if face_encodings == []:
        msg = "You are unknown, first regiter yourself"
        status = "You are unknown"
    else:
        for face_encoding in face_encodings:
            # Comparo las caras
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding, tolerance=0.55)
            name = "Unknown"
            # Hallo distancia de las caras, se usa ecuación distancia-punto
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                set_out_time(known_rfids[best_match_index])

            if name != "Unknown":
                msg = name
                status = "ok"
            else:
                msg = "You are unknown, first register yourself"
                status = "You are unknown"
                face_names.append(name)

            rand_no = np.random.random_sample()
            cv2.imwrite(unknown_path+str(rand_no)+".jpg", frame)

    return JSONResponse(content={"data": msg, "status": status}, media_type="application/json")

# With rfid


@app.get(path="/logout-with-rfid", summary="Logout with RfId", tags=["Logout"])
def logoutRfId(rfid: Optional[str]):
    get_data()
    global db

    if db == []:
        msg = "You are unknown, first register yourself"
        status = "Error"
    else:

        try:
            known_rfids = [i[6] for i in db]
            known_names = [i[1] for i in db]
            set_out_time(rfid)

            idx = -1
            for i in range(known_rfids):

                if known_rfids[i] == rfid:
                    idx = i

            if idx != -1:
                msg = known_names[idx]
                status = "ok"
            else:
                msg = "Rfid unknown"
                status = "error"
        except:
            return JSONResponse({"status": "error"})

    return JSONResponse(content={"data": msg, "status": status}, media_type="application/json")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)