# Sistema de recomendación de canciones

Sistema recomendador de canciones

1. Activar entorno virtual (opcional)
```shell
conda create -n p2_bd2 python=3.10 -y
```

```shell
conda activate p2_bd2
```

2. Instalar dependencias

```shell
pip install -r requirements.txt
```

3. Levantar API backend

```shell
uvicorn backend.main:app --reload
```

4. Levantar app (en simultaneo con API backend)

```shell
streamlit run frontend/app.py
```

## Pruebas

Para ejecutar scripts de prueba

1. Para índice invertido
```shell
python -m test.build_index
```
2. Para base de datos multimedia

```shell
python -m test.test_audio_index 
```
3. Para construir índice multimedia

```shell
python -m test.build_audio_index
```


## Diagrama de flujo de aplicación

![alt text](diagrama_app.png)