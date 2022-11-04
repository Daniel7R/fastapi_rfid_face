# Software Architecture - Challenge 2

This is built with FastAPI, Open-cv, Face-Recognition and PostgreSQL.

- At the first, Flask is used to create the api for the users requests
- Open Cv is used to access to the camera and take the photos to save in the db or log into the app with face-recognition
- Face recognition is a library that constains multiple face recognition functions, some of them were used in this project, for example, compare_faces, face_encodings and face_distances
- Finally and not less important PostgreSQL that is db motor where we are persisting the users data

The database must have the followgin schema or table:

```
CREATE TABLE IF NOT EXISTS public.usuarios
(
    id character varying(15) COLLATE pg_catalog."default" NOT NULL,
    nombres character varying(60) COLLATE pg_catalog."default",
    edad integer,
    genero character varying(15) COLLATE pg_catalog."default",
    estrato integer,
    departamento character varying(30) COLLATE pg_catalog."default",
    rfid character varying(20) COLLATE pg_catalog."default",
    in_time timestamptz,
    out_time timestamptz,
    accumulator real,
    face text COLLATE pg_catalog."default",
    CONSTRAINT usuarios_pkey PRIMARY KEY (id),
    CONSTRAINT usuarios_rfid_key UNIQUE (rfid)
)
```
