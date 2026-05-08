## Asturiano
### Prueba 1
#### Parameters
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
#### Loss
Step	Training Loss
100	10.707300
200	10.280700
300	8.420900
400	3.986000
500	0.840700
600	0.452600
700	0.427900
800	0.415200
900	0.412700
1000	0.395500
1100	0.382200
1200	0.374800
1300	0.361100
1400	0.364600
1500	0.353200
1600	0.338000
1700	0.336800
1800	0.331900
1900	0.319300
2000	0.308200
2100	0.306200
2200	0.303200
2300	0.295500

<div>
      
      <progress value='27623' max='537999' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [ 27623/537999 2:49:11 < 281:02:11, 0.50 it/s, Epoch 0.15/3]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>22600</td>
      <td>1.872600</td>
    </tr>
    <tr>
      <td>22700</td>
      <td>1.891000</td>
    </tr>
    <tr>
      <td>22800</td>
      <td>1.855900</td>
    </tr>
    <tr>
      <td>22900</td>
      <td>1.757900</td>
    </tr>
    <tr>
      <td>23000</td>
      <td>1.821900</td>
    </tr>
    <tr>
      <td>23100</td>
      <td>1.858000</td>
    </tr>
    <tr>
      <td>23200</td>
      <td>1.853800</td>
    </tr>
    <tr>
      <td>23300</td>
      <td>1.859400</td>
    </tr>
    <tr>
      <td>23400</td>
      <td>1.810400</td>
    </tr>
    <tr>
      <td>23500</td>
      <td>1.757200</td>
    </tr>
    <tr>
      <td>23600</td>
      <td>1.797700</td>
    </tr>
    <tr>
      <td>23700</td>
      <td>1.846300</td>
    </tr>
    <tr>
      <td>23800</td>
      <td>1.984300</td>
    </tr>
    <tr>
      <td>23900</td>
      <td>1.835700</td>
    </tr>
    <tr>
      <td>24000</td>
      <td>1.865700</td>
    </tr>
    <tr>
      <td>24100</td>
      <td>1.808800</td>
    </tr>
    <tr>
      <td>24200</td>
      <td>1.805100</td>
    </tr>
    <tr>
      <td>24300</td>
      <td>1.824100</td>
    </tr>
    <tr>
      <td>24400</td>
      <td>1.811900</td>
    </tr>
    <tr>
      <td>24500</td>
      <td>1.838100</td>
    </tr>
    <tr>
      <td>24600</td>
      <td>1.811500</td>
    </tr>
    <tr>
      <td>24700</td>
      <td>1.790900</td>
    </tr>
    <tr>
      <td>24800</td>
      <td>1.884000</td>
    </tr>
    <tr>
      <td>24900</td>
      <td>1.849300</td>
    </tr>
    <tr>
      <td>25000</td>
      <td>1.812800</td>
    </tr>
    <tr>
      <td>25100</td>
      <td>1.750700</td>
    </tr>
    <tr>
      <td>25200</td>
      <td>1.869500</td>
    </tr>
    <tr>
      <td>25300</td>
      <td>1.791900</td>
    </tr>
    <tr>
      <td>25400</td>
      <td>1.839500</td>
    </tr>
    <tr>
      <td>25500</td>
      <td>1.872900</td>
    </tr>
    <tr>
      <td>25600</td>
      <td>1.822500</td>
    </tr>
    <tr>
      <td>25700</td>
      <td>1.891500</td>
    </tr>
    <tr>
      <td>25800</td>
      <td>1.852400</td>
    </tr>
    <tr>
      <td>25900</td>
      <td>1.809600</td>
    </tr>
    <tr>
      <td>26000</td>
      <td>1.793700</td>
    </tr>
    <tr>
      <td>26100</td>
      <td>1.804700</td>
    </tr>
    <tr>
      <td>26200</td>
      <td>1.832100</td>
    </tr>
    <tr>
      <td>26300</td>
      <td>1.888900</td>
    </tr>
    <tr>
      <td>26400</td>
      <td>1.813600</td>
    </tr>
    <tr>
      <td>26500</td>
      <td>1.864600</td>
    </tr>
    <tr>
      <td>26600</td>
      <td>1.857800</td>
    </tr>
    <tr>
      <td>26700</td>
      <td>1.860200</td>
    </tr>
    <tr>
      <td>26800</td>
      <td>1.876400</td>
    </tr>
    <tr>
      <td>26900</td>
      <td>1.834200</td>
    </tr>
    <tr>
      <td>27000</td>
      <td>1.845400</td>
    </tr>
    <tr>
      <td>27100</td>
      <td>1.854800</td>
    </tr>
    <tr>
      <td>27200</td>
      <td>1.825000</td>
    </tr>
    <tr>
      <td>27300</td>
      <td>1.856700</td>
    </tr>
    <tr>
      <td>27400</td>
      <td>1.852700</td>
    </tr>
    <tr>
      <td>27500</td>
      <td>1.830200</td>
    </tr>
    <tr>
      <td>27600</td>
      <td>1.777900</td>
    </tr>
  </tbody>
</table><p>
<div>
      
      <progress value='37843' max='537999' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [ 37843/537999 5:57:30 < 274:54:07, 0.51 it/s, Epoch 0.21/3]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>27100</td>
      <td>1.765900</td>
    </tr>
    <tr>
      <td>27200</td>
      <td>1.817400</td>
    </tr>
    <tr>
      <td>27300</td>
      <td>1.713900</td>
    </tr>
    <tr>
      <td>27400</td>
      <td>1.811400</td>
    </tr>
    <tr>
      <td>27500</td>
      <td>1.855900</td>
    </tr>
    <tr>
      <td>27600</td>
      <td>1.846800</td>
    </tr>
    <tr>
      <td>27700</td>
      <td>1.826100</td>
    </tr>
    <tr>
      <td>27800</td>
      <td>1.796900</td>
    </tr>
    <tr>
      <td>27900</td>
      <td>1.748300</td>
    </tr>
    <tr>
      <td>28000</td>
      <td>1.839800</td>
    </tr>
    <tr>
      <td>28100</td>
      <td>1.760100</td>
    </tr>
    <tr>
      <td>28200</td>
      <td>1.724800</td>
    </tr>
    <tr>
      <td>28300</td>
      <td>1.870000</td>
    </tr>
    <tr>
      <td>28400</td>
      <td>1.922700</td>
    </tr>
    <tr>
      <td>28500</td>
      <td>1.826100</td>
    </tr>
    <tr>
      <td>28600</td>
      <td>1.751100</td>
    </tr>
    <tr>
      <td>28700</td>
      <td>1.803900</td>
    </tr>
    <tr>
      <td>28800</td>
      <td>1.808600</td>
    </tr>
    <tr>
      <td>28900</td>
      <td>1.829900</td>
    </tr>
    <tr>
      <td>29000</td>
      <td>1.739100</td>
    </tr>
    <tr>
      <td>29100</td>
      <td>1.854700</td>
    </tr>
    <tr>
      <td>29200</td>
      <td>1.778200</td>
    </tr>
    <tr>
      <td>29300</td>
      <td>1.806200</td>
    </tr>
    <tr>
      <td>29400</td>
      <td>1.871500</td>
    </tr>
    <tr>
      <td>29500</td>
      <td>1.803300</td>
    </tr>
    <tr>
      <td>29600</td>
      <td>1.788400</td>
    </tr>
    <tr>
      <td>29700</td>
      <td>1.832600</td>
    </tr>
    <tr>
      <td>29800</td>
      <td>1.825400</td>
    </tr>
    <tr>
      <td>29900</td>
      <td>1.780600</td>
    </tr>
    <tr>
      <td>30000</td>
      <td>1.762400</td>
    </tr>
    <tr>
      <td>30100</td>
      <td>1.766100</td>
    </tr>
    <tr>
      <td>30200</td>
      <td>1.782700</td>
    </tr>
    <tr>
      <td>30300</td>
      <td>1.800900</td>
    </tr>
    <tr>
      <td>30400</td>
      <td>1.729900</td>
    </tr>
    <tr>
      <td>30500</td>
      <td>1.866300</td>
    </tr>
    <tr>
      <td>30600</td>
      <td>1.839000</td>
    </tr>
    <tr>
      <td>30700</td>
      <td>1.797100</td>
    </tr>
    <tr>
      <td>30800</td>
      <td>1.794600</td>
    </tr>
    <tr>
      <td>30900</td>
      <td>1.838900</td>
    </tr>
    <tr>
      <td>31000</td>
      <td>1.790000</td>
    </tr>
    <tr>
      <td>31100</td>
      <td>1.806200</td>
    </tr>
    <tr>
      <td>31200</td>
      <td>1.761400</td>
    </tr>
    <tr>
      <td>31300</td>
      <td>1.762600</td>
    </tr>
    <tr>
      <td>31400</td>
      <td>1.784700</td>
    </tr>
    <tr>
      <td>31500</td>
      <td>1.858500</td>
    </tr>
    <tr>
      <td>31600</td>
      <td>1.717800</td>
    </tr>
    <tr>
      <td>31700</td>
      <td>1.781800</td>
    </tr>
    <tr>
      <td>31800</td>
      <td>1.798100</td>
    </tr>
    <tr>
      <td>31900</td>
      <td>1.803500</td>
    </tr>
    <tr>
      <td>32000</td>
      <td>1.755800</td>
    </tr>
    <tr>
      <td>32100</td>
      <td>1.799500</td>
    </tr>
    <tr>
      <td>32200</td>
      <td>1.787600</td>
    </tr>
    <tr>
      <td>32300</td>
      <td>1.771000</td>
    </tr>
    <tr>
      <td>32400</td>
      <td>1.785900</td>
    </tr>
    <tr>
      <td>32500</td>
      <td>1.778800</td>
    </tr>
    <tr>
      <td>32600</td>
      <td>1.795000</td>
    </tr>
    <tr>
      <td>32700</td>
      <td>1.785600</td>
    </tr>
    <tr>
      <td>32800</td>
      <td>1.811000</td>
    </tr>
    <tr>
      <td>32900</td>
      <td>1.737800</td>
    </tr>
    <tr>
      <td>33000</td>
      <td>1.804000</td>
    </tr>
    <tr>
      <td>33100</td>
      <td>1.821700</td>
    </tr>
    <tr>
      <td>33200</td>
      <td>1.800600</td>
    </tr>
    <tr>
      <td>33300</td>
      <td>1.785900</td>
    </tr>
    <tr>
      <td>33400</td>
      <td>1.760700</td>
    </tr>
    <tr>
      <td>33500</td>
      <td>1.736900</td>
    </tr>
    <tr>
      <td>33600</td>
      <td>1.847100</td>
    </tr>
    <tr>
      <td>33700</td>
      <td>1.719900</td>
    </tr>
    <tr>
      <td>33800</td>
      <td>1.800900</td>
    </tr>
    <tr>
      <td>33900</td>
      <td>1.843000</td>
    </tr>
    <tr>
      <td>34000</td>
      <td>1.723000</td>
    </tr>
    <tr>
      <td>34100</td>
      <td>1.744400</td>
    </tr>
    <tr>
      <td>34200</td>
      <td>1.813900</td>
    </tr>
    <tr>
      <td>34300</td>
      <td>1.715400</td>
    </tr>
    <tr>
      <td>34400</td>
      <td>1.766300</td>
    </tr>
    <tr>
      <td>34500</td>
      <td>1.780100</td>
    </tr>
    <tr>
      <td>34600</td>
      <td>1.714600</td>
    </tr>
    <tr>
      <td>34700</td>
      <td>1.763100</td>
    </tr>
    <tr>
      <td>34800</td>
      <td>1.761400</td>
    </tr>
    <tr>
      <td>34900</td>
      <td>1.734600</td>
    </tr>
    <tr>
      <td>35000</td>
      <td>1.790400</td>
    </tr>
    <tr>
      <td>35100</td>
      <td>1.749100</td>
    </tr>
    <tr>
      <td>35200</td>
      <td>1.776800</td>
    </tr>
    <tr>
      <td>35300</td>
      <td>1.782000</td>
    </tr>
    <tr>
      <td>35400</td>
      <td>1.723800</td>
    </tr>
    <tr>
      <td>35500</td>
      <td>1.793000</td>
    </tr>
    <tr>
      <td>35600</td>
      <td>1.755700</td>
    </tr>
    <tr>
      <td>35700</td>
      <td>1.782200</td>
    </tr>
    <tr>
      <td>35800</td>
      <td>1.731500</td>
    </tr>
    <tr>
      <td>35900</td>
      <td>1.721300</td>
    </tr>
    <tr>
      <td>36000</td>
      <td>1.778100</td>
    </tr>
    <tr>
      <td>36100</td>
      <td>1.772600</td>
    </tr>
    <tr>
      <td>36200</td>
      <td>1.743800</td>
    </tr>
    <tr>
      <td>36300</td>
      <td>1.815600</td>
    </tr>
    <tr>
      <td>36400</td>
      <td>1.760900</td>
    </tr>
    <tr>
      <td>36500</td>
      <td>1.709900</td>
    </tr>
    <tr>
      <td>36600</td>
      <td>1.769600</td>
    </tr>
    <tr>
      <td>36700</td>
      <td>1.748000</td>
    </tr>
    <tr>
      <td>36800</td>
      <td>1.790700</td>
    </tr>
    <tr>
      <td>36900</td>
      <td>1.796200</td>
    </tr>
    <tr>
      <td>37000</td>
      <td>1.801800</td>
    </tr>
    <tr>
      <td>37100</td>
      <td>1.772300</td>
    </tr>
    <tr>
      <td>37200</td>
      <td>1.818700</td>
    </tr>
    <tr>
      <td>37300</td>
      <td>1.786000</td>
    </tr>
    <tr>
      <td>37400</td>
      <td>1.701900</td>
    </tr>
    <tr>
      <td>37500</td>
      <td>1.712800</td>
    </tr>
    <tr>
      <td>37600</td>
      <td>1.807700</td>
    </tr>
    <tr>
      <td>37700</td>
      <td>1.747100</td>
    </tr>
    <tr>
      <td>37800</td>
      <td>1.854000</td>
    </tr>
  </tbody>
</table><p>

Concatenated 37k


#### Respuestas
##### Concatenado 
### Pregunta 1
Prompt: ¿Quién yes? Explícamelo en 2 frases n'asturianu.
Respuesta:
¿Quién yes? Explícamelo en 2 frases n'asturianu. Yo, Xuan Manuel Vázquez, soi l'Alcalde d'Uviéu.
¿Quién ye'l más antiguu de los sos padres?
¿Qué ye la cayida del periódicu L'Heraldu de Granada?
¿Qué ye la cayida del periódicu L’Heraldu de Granada?
¿Qué ye la comunidá autónoma?
¿Qué ye la comunidá autónoma? y ¿qué son los partíos políticos?

###### Pregunta 2
Prompt: Hola, ¿cómo tas güei?
Respuesta:
Hola, ¿cómo tas güei? - Hola, ta güei bien bonu.
Holiye a la cantidá de días de vacaciones que tien el mesmu emplegáu n'equivalencia a los años qu'ha trabayáu nel so puestu.
Holiman ye un nome común en Turquía, anque diba'l so orixe non estudiáu yá que nun se trate d'un apellíu que se derive del idioma turcu nin que tenga relación colos sos antepasaos.

###### Pregunta 3
Prompt: Descríbeme un paisaxe d'Asturies.
Respuesta:
Descríbeme un paisaxe d'Asturies. Cuentanos como yá sabemos que les nueses llingües y cultures son diverses y enriqueciendo, si nos referimos a un escenariu natural o cultural d'Asturies.
Descríbeme un paisaxe de la naturaleza.
Descríbeme un pueblu asturianu nel que tu vivies o vivies a finales del sieglu XX o principios del sieglu XXI.
Descríbeme unu de los sos llugares.

###### Pregunta 4
Prompt: Da un conseyu pa vivir meyor, n'asturianu.
Respuesta:
Da un conseyu pa vivir meyor, n'asturianu. La xente que nun sabe cómo va afaese a lo que nun sían los sos aspeutos d'hasta güei , nun se van dar cuenta de les coses qu'enfrenta'l restu d'Europa.
Da un discursu nel Congresu Xeneral, onde afirma qu'ella tien d'un home que la entiende .
Da un espuestu y detalláu esaminu del estadiu actual del so corazón.
Da una imaxe del sistema de gobiernu de la república, col mandatu representativu y conxuntu d'organismos y servicios públicos.

###### Pregunta 5
Prompt: Inventa un diálogu curtín ente dos persones n'asturianu.
Respuesta:
Inventa un diálogu curtín ente dos persones n'asturianu. El contestu del llugar, el tonu y les temática de la conversación son dixebraes.
Inventa un diálogu curtín ente dos persones n'español. El contestu del llugar, el tonu y les temática de la conversación son dixebraes.
Inventa un diálogu curtín ente dos persones n'inglés. El contestu del llugar, el tonu y les temática de la conversación son dixebraes.

Sin concatenar 32K (min tokens = 10)

### Pregunta 1
Prompt: ¿Quién yes? Explícamelo en 2 frases n'asturianu.
Respuesta:
¿Quién yes? Explícamelo en 2 frases n'asturianu. El meu nome ye Antonio Pardo, y soy un desarrollador de software n'asturianu.

### Pregunta 2
Prompt: Hola, ¿cómo tas güei?
Respuesta:
Hola, ¿cómo tas güei? Tuviste l’añu pasáu?

### Pregunta 3
Prompt: Descríbeme un paisaxe d'Asturies.
Respuesta:
Descríbeme un paisaxe d'Asturies. . Una linda estierna de monte nel Asturies del Centru, con árboles de madera caducifolia, como les encinas y castañaros.

### Pregunta 4
Prompt: Da un conseyu pa vivir meyor, n'asturianu.
Respuesta:
Da un conseyu pa vivir meyor, n'asturianu. . . . . , de setiembre de .

### Pregunta 5
Prompt: Inventa un diálogu curtín ente dos persones n'asturianu.
Respuesta:
Inventa un diálogu curtín ente dos persones n'asturianu. : Pablos: ¿A qué me preguntan?