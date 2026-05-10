# TFG
TFG about LLM and minoritary spanish languages

## Herramientas
https://github.com/TALP-UPC/FreeLing/tree/master

## Pasos a realizar
### Preparar para entregar
- [ ] Leerse la memoria completa para buscar errores y comprobar que está todo bien
- [ ] Limpiar el código (debug en init, read_local_file en LanguageDataset, las funciones de generardatasets)
- [ ] Comprobar que el código está bien todo, también notebooks
- [ ] Subir modelos, datasets y código a hf, pip...
- [ ] Subir memoria a GitHub, en latex y pdf
### Memoria
- [x] Cambiar color de descripciones en gráfico
- [x] Como se genera, limpian los datos en 4.2 Materiales
- [x] De donde vienen los datos en 4.2 Materiales
- [x] Añadir pequeña explicacion de Bleue y chrF
- [x] Citar de donde vienen los lexicon -> FreeLing
- [x] Añadir figura en 4.2.2 (diagrama...)
- [x] Identificar posibles soluciones (PEFT, Full fine tuning, algo más?) => explicar olvido catastrófico
L A revisar
- [x] Actualizar en la memoria los modelos elegidos y por qué (después de elegir los modelos)
L A revisar
- [x] Escribir resultados en memoria
- [x] Revisar aspectos legales y eticos (8) y metodología (4.1)
- [x] Redactar sección 1
L A revisar
- [x] Explicar las capas de los LLM en el Marco teórico
L A revisar
- [ ] Añadir estadísticas de datasets elegidos y generados
- [ ] Poner perdidas loss de los entrenamientos en 6.3
- [ ] Terminar de escribir pruebas realizadas.
- [ ] Comentar en el 6.3 los test realizados con asturiano y entrenar igual con gallego y aranes, sacar resultados y ponerlos
- [ ] Revisar 6.2 y 6.3
- [ ] Continuar con 6.3
- [ ] Escribir 7.2 y 7.3
- [x] Comentar pad token
L Revisar
- [x] Explicar bien a que capas vamos a aplicarlo
L Revisarlo
- [x] Contar primer entrenamiento que devolvía siempre EOS al principio
- [x] Poner que vamos a reocrtar en el percentil 95, para evitar tanto EOS
- [ ] Comentar los resultados
- [ ] Reducir entonces resultados no interesantes
- [ ] Conclusiones y Future Work
- [ ] En future work, añadir que para cortar bien igual sería mejor cortar con un llm (no me da el cómputo)

### Datos
- [ ] Crear dataset para Gallego, Asturiano y  Aranés/Occitano [Train]
- [x] Crear lexicons
- [x] Crear datasets anotados

### Benchmark
- [x] Elegir métricas (Benchmark debajo) => No he encontrado papers sobre esto
- [x] Implementar Benchmark
- [x] Probar Benchmark con Dataset Anotado


### Entrenamiento
- [x] Elegir modelos más grandes y ver que se pueden cuantizar y cargar
- [x] Crear esqueleto QLoRA entrenamiento
- [ ] Entrenar con LoRA sobre el mejor 

### Evaluación
- [x] Evaluar los modelos base Asturiano
- [x] Evaluar los modelos base Gallego
- [x] Evaluar los modelos base Aranés
- [ ] Evaluar resultados QLoRA Asturiano
- [ ] Evaluar resultados QLoRA Gallego
- [ ] Evaluar resultados QLoRA Aranés