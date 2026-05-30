# TFG
TFG about LLM and minoritary spanish languages

## Enlaces
[Freeling](https://github.com/TALP-UPC/FreeLing/tree/master)
[Diagrama Limpieza](https://lucid.app/lucidchart/e8872fd3-94e5-4f96-99c3-397eefb902df/edit?view_items=U4FrEd7CHWG8%2CU4Fr2y0hQgMb%2CqaGrOuOa6h2R%2CZ~Fr3DKrELBR%2CU4FrM8M~b0xO%2CBeGrPuh-.JKh%2C1bGrU8zP9B9f%2CU4FrKWUGB.N5%2CrbGrTWlpsdON%2CCcGrdZRyF.Ug%2CJbGrR4L2g5pi%2CFeGrEE~~BHZM%2CU4Fr~Kr5BzzA%2CU4FrmymcKsFr%2CsdGryhPBCqdJ%2CEkGrVK~5b.8k%2CkiGrPXrDpMC-%2CxjGrwdmI5.DL%2CV4FruWAmYfkM%2CJ-Frbo1-ceBi%2CV4FracIGt3Hq%2CV4FrTVfRWT_R%2CV4FrLMgxxX71&page=0_0&invitationId=inv_87f05792-3574-4ccd-81cb-d6179ddb62f2)
[Diagrama metodología](https://whimsical.com/tfg4685/Reg2dMWQGdewbjpgnDLLq8)

## Pasos a realizar
### Preparar para entregar
- [ ] Leerse la memoria completa para buscar errores y comprobar que está todo bien
- [ ] Limpiar el código (debug en init, read_local_file en LanguageDataset, las funciones de generardatasets)
- [ ] Comprobar que el código está bien todo, también notebooks
- [ ] Subir modelos, datasets y código a hf, pip...
- [ ] Subir memoria a GitHub, en latex y pdf
- [ ] Comprobar que no se usa 1 persona en el 6
### Memoria
- [x] Cambiar color de descripciones en gráfico
- [x] Como se genera, limpian los datos en 4.2 Materiales
- [x] De donde vienen los datos en 4.2 Materiales
- [x] Añadir pequeña explicacion de Bleue y chrF
- [x] Citar de donde vienen los lexicon -> FreeLing
- [x] Añadir figura en 4.2.2 (diagrama...)
- [x] Identificar posibles soluciones (PEFT, Full fine tuning, algo más?) => explicar olvido catastrófico
- [x] Actualizar en la memoria los modelos elegidos y por qué (después de elegir los modelos)
- [x] Escribir resultados en memoria
- [x] Revisar aspectos legales y eticos (8) y metodología (4.1)
- [x] Redactar sección 1
- [x] Explicar las capas de los LLM en el Marco teórico
- [ ] Añadir estadísticas de datasets elegidos y generados
- [ ] Explicar sobre losses de los entrenamientos en 6.3
- [ ] Terminar de escribir pruebas realizadas.
- [ ] Comentar en el 6.3 los test realizados con asturiano y entrenar igual con gallego y aranes, sacar resultados y ponerlos
- [ ] Revisar 6.2 y 6.3
- [ ] Continuar con 6.3
- [ ] Escribir 7.2 y 7.3
- [x] Comentar pad token
- [x] Explicar bien a que capas vamos a aplicarlo
- [x] Contar primer entrenamiento que devolvía siempre EOS al principio
- [x] Poner que vamos a reocrtar en el percentil 95, para evitar tanto EOS
- [ ] Comentar los resultados
- [ ] Reducir entonces resultados no interesantes
- [ ] Conclusiones y Future Work
- [ ] En future work, añadir que para cortar bien igual sería mejor cortar con un llm (no me da el cómputo)

### Datos
- [x] Crear dataset para Gallego, Asturiano y  Aranés/Occitano \[Train\]
- [x] Crear lexicons
- [x] Crear datasets anotados

### Benchmark
- [x] Elegir métricas (Benchmark debajo) => No he encontrado papers sobre esto
- [x] Implementar Benchmark
- [x] Probar Benchmark con Dataset Anotado


### Entrenamiento
- [x] Elegir modelos más grandes y ver que se pueden cuantizar y cargar
- [x] Crear esqueleto QLoRA entrenamiento
- [x] Entrenar con QLoRA sobre el mejor 

### Evaluación
- [x] Evaluar los modelos base Asturiano
- [x] Evaluar los modelos base Gallego
- [x] Evaluar los modelos base Aranés
- [ ] Evaluar resultados QLoRA Asturiano
- [ ] Evaluar resultados QLoRA Gallego
- [ ] Evaluar resultados QLoRA Aranés