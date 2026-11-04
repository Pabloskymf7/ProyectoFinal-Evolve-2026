# Actividad 0 — Equipos, ideas de producto y repositorio

**Máster:** Evolve · Máster de Data Science e IA · 2026–2027  
**Estado:** Exploración de ideas  
**Repositorio:** [ProyectoFinal-Evolve-2026](https://github.com/Pabloskymf7/ProyectoFinal-Evolve-2026)


## Composición del equipo

| Integrante | Usuario de GitHub |
|------------|-------------------|
| [Pablo Miguel Ferrer] | [@Pabloskymf7] |
| [Jorge Martinez Cercadillo] | [@Jmcercadillo3] |
| [Julen Asua Fullaondo] | [@fuyacck] |
| [Joan Poveda Baixauli] | [@JoanP21] |

**Idea preferida** provisional: HousingPulse AI, porque dispone de una primera fuente de datos inmobiliarios descargada y permite desarrollar un modelo predictivo con una aplicación web funcional. El proyecto se centrará inicialmente en el condado de Miami-Dade (Florida), lo que permitirá trabajar con un ámbito geográfico delimitado y analizar las diferencias de precios entre zonas. Además, ofrece posibilidades de ampliación y mejora a partir de nuevos datos y funcionalidades. La elección definitiva queda pendiente de comprobar la calidad y representatividad de los datos, contrastar la necesidad con usuarios y recibir feedback del profesorado.

---

# Idea 1. HousingPulse AI — Estimación inteligente de precios inmobiliarios

## Título y descripción

HousingPulse AI es una plataforma web que utiliza técnicas de Machine Learning para estimar el precio de venta de una vivienda a partir de sus características físicas y su ubicación geográfica.

El usuario podrá introducir información como la ciudad, el código postal, la superficie, el número de habitaciones y baños, el año de construcción y el tipo de inmueble. A partir de estos datos, el sistema generará una estimación orientativa del precio de la vivienda, acompañada de información sobre los factores que más influyen en el resultado.

El proyecto se centrará inicialmente en el condado de Miami-Dade, Florida, y combinará análisis exploratorio de datos, preparación de datos inmobiliarios, entrenamiento y evaluación de modelos predictivos y desarrollo de una interfaz web interactiva.

## Problema y usuario

Conocer el valor aproximado de una vivienda puede resultar complicado, ya que el precio depende de múltiples factores, como la ubicación, la superficie, el estado del inmueble y las características del mercado inmobiliario de cada zona.

Las herramientas existentes pueden ofrecer estimaciones generales, pero no siempre explican qué variables influyen en el resultado ni permiten conocer las limitaciones de la predicción.

Actualmente, los usuarios suelen consultar portales inmobiliarios, comparar viviendas similares o recurrir a profesionales del sector para obtener una valoración. Sin embargo, estas alternativas pueden requerir comparaciones manuales y no siempre ofrecen una explicación clara de los factores que influyen en el precio.

HousingPulse AI pretende facilitar una primera estimación orientativa mediante un modelo entrenado con datos inmobiliarios históricos recientes.

Los usuarios iniciales serían particulares interesados en conocer el valor aproximado de su vivienda, compradores que quieran comparar inmuebles y profesionales inmobiliarios que necesiten una herramienta complementaria para analizar propiedades.

## Uso y valor

Un usuario quiere conocer el precio aproximado de su vivienda en Miami-Dade. Para ello, introduce su código postal, la superficie, el número de habitaciones y baños, el año de construcción y otras características disponibles.

La plataforma procesa esta información y devuelve una estimación del precio de venta, mostrando los principales factores que han influido en la predicción y, cuando sea posible, una referencia de viviendas comparables.

El valor principal del proyecto sería transformar datos inmobiliarios en una estimación accesible y comprensible, reduciendo la necesidad de realizar comparaciones manuales.

La estimación se presentaría como una referencia orientativa, no como una tasación oficial ni como una garantía del precio final de compraventa.

## Datos y fuentes candidatas

- **RealEstateDataset — datos inmobiliarios de Estados Unidos:** [https://realestatedataset.com/](https://realestatedataset.com/)

  Fuente del conjunto de datos descargado para el proyecto. El archivo seleccionado corresponde al condado de Miami-Dade, Florida, y contiene información sobre propiedades individuales, ubicación, características físicas, precios y fechas de última venta.

- **Archivo de trabajo:** `realestatedataset-miami-dade-county-fl-2026-10.csv`.

  El archivo descargado contiene aproximadamente un millón de registros y ocupa unos 162 MB. Entre sus variables se encuentran `city`, `zip`, `lat`, `lng`, `property_type`, `beds`, `baths`, `sqft`, `year_built`, `list_price`, `last_sold_price` y `last_sold_date`.

- **Documentación de la fuente:** [https://realestatedataset.com/methodology](https://realestatedataset.com/methodology)

  Se utilizará para comprobar el significado de las variables, el procedimiento de recopilación de datos, las limitaciones de la información y las condiciones de uso.

**Estado de acceso:** se ha descargado un archivo inmobiliario correspondiente a Miami-Dade y se ha realizado una primera revisión de su estructura y de los valores ausentes. La selección definitiva de registros dependerá de la comprobación de la calidad de los precios, las fechas y las características de las viviendas.

La primera exploración identifica decenas de miles de registros con fecha de última venta desde 2025. No obstante, será necesario verificar que esas fechas correspondan a operaciones adecuadas para el objetivo predictivo y que los precios asociados sean fiables. También se comprobará si existen registros duplicados o propiedades cuya información se haya actualizado en diferentes momentos.

**Variable objetivo:** como primera aproximación, se estudiará `last_sold_price` como variable objetivo del modelo, siempre que se pueda verificar su significado y la fiabilidad de los registros. Esta variable representa el último precio de venta registrado y no debe confundirse con `list_price`, que corresponde al precio anunciado. La variable `list_price` podrá utilizarse en análisis complementarios, pero no se mezclarán ambos conceptos en el entrenamiento del modelo.

**Privacidad y restricciones:** se revisarán las condiciones de uso y redistribución de la fuente antes de publicar los datos o desplegar la aplicación. Se evitará exponer información innecesaria sobre direcciones particulares y se documentarán las fuentes, las transformaciones realizadas y las limitaciones del conjunto de datos.

## Primera versión o MVP

1. **Auditoría y limpieza de datos:** analizar valores ausentes, duplicados, registros inconsistentes y posibles valores extremos. Seleccionar las viviendas con información suficiente y precios válidos.
2. **Análisis exploratorio:** estudiar la distribución de los precios y su relación con la superficie, las habitaciones, los baños, la antigüedad y la ubicación.
3. **Preparación de variables:** transformar las variables categóricas, gestionar los valores ausentes y preparar las características para el entrenamiento.
4. **Desarrollo de modelos:** construir un modelo base de regresión y compararlo con algoritmos como Random Forest y Gradient Boosting.
5. **Evaluación:** medir el rendimiento con métricas como MAE y RMSE. Siempre que la calidad de las fechas lo permita, se utilizará una separación temporal para evaluar el modelo con registros más recientes que los utilizados durante el entrenamiento, evitando fugas de información.
6. **Interpretabilidad:** analizar qué variables influyen más en las estimaciones y estudiar cómo cambia el precio previsto según las características del inmueble.
7. **Aplicación web:** desarrollar una interfaz con Streamlit en la que el usuario introduzca las características de una vivienda y obtenga una estimación.
8. **Informe final:** documentar los datos utilizados, el proceso de limpieza, los modelos comparados, los resultados, las limitaciones y las posibles mejoras.

El principal criterio para valorar el éxito del modelo será el error medio de las estimaciones, medido mediante MAE y RMSE. Se estudiará también si el error es similar entre distintas zonas y tipos de vivienda. El objetivo será conseguir estimaciones suficientemente fiables para ofrecer una referencia orientativa, sin establecer de antemano un nivel de precisión que todavía no se ha comprobado.

## Fuera del MVP

Quedarían fuera de la primera versión la tasación oficial, las recomendaciones de inversión, la predicción garantizada de precios futuros, la cobertura de todo Estados Unidos y la integración con plataformas inmobiliarias externas en tiempo real.

Tampoco se presentarán como precios actuales estimaciones obtenidas exclusivamente a partir de operaciones antiguas sin una evaluación específica de su capacidad para representar el mercado reciente.

## Viabilidad y dudas

**Principal dificultad:** conseguir que los datos utilizados para entrenar el modelo representen adecuadamente el mercado inmobiliario actual. Aunque el archivo contiene registros recientes, será necesario comprobar la validez de las fechas, la fiabilidad de los precios y la cantidad de operaciones disponibles en cada zona.

Otra dificultad será la heterogeneidad de las propiedades. Los precios de casas unifamiliares, apartamentos y otros tipos de inmuebles pueden responder a factores diferentes. También será necesario evaluar si el modelo funciona de forma similar en distintas ciudades y códigos postales dentro de Miami-Dade.

**Qué comprobar:** verificar el significado de las variables de precio y fecha, identificar posibles duplicados, analizar la distribución geográfica de las operaciones y revisar los valores extremos. También se comprobará que las variables utilizadas para predecir el precio estén disponibles en el momento en el que un usuario realiza la consulta.

Después se construirá un modelo base y se compararán distintos algoritmos mediante métricas de error. Se estudiará si la muestra permite realizar predicciones fiables en las zonas seleccionadas y se documentarán los casos en los que la información disponible sea insuficiente.

**Valor diferencial:** HousingPulse AI no se limitará a mostrar estadísticas inmobiliarias, sino que desarrollará un modelo predictivo propio y una aplicación web funcional para estimar el precio de viviendas individuales. El proyecto combinará Machine Learning aplicado, análisis de datos, evaluación rigurosa y desarrollo de producto, con especial atención a la actualidad y la calidad de los datos.
---

# Idea 2. GoalMetrics AI — Plataforma de analítica avanzada del rendimiento en fútbol mediante Expected Goals (xG)

## Título y descripción

Desarrollo de una herramienta de analítica deportiva capaz de estimar la calidad de las ocasiones de gol en fútbol mediante un modelo de aprendizaje automático de Expected Goals (xG).

El sistema analizaría los eventos de disparo de los partidos y calcularía la probabilidad de que cada lanzamiento termine en gol, teniendo en cuenta variables como la distancia y el ángulo respecto a la portería, la parte del cuerpo utilizada, el tipo de disparo y el contexto de la jugada.

A partir de estas estimaciones, la plataforma permitiría analizar el rendimiento ofensivo de equipos y jugadores, comparando los goles reales con los esperados y ofreciendo información más completa que las estadísticas tradicionales.

## Problema y usuario

El número de goles o disparos no siempre refleja correctamente el rendimiento ofensivo de un equipo o jugador. Dos equipos pueden realizar el mismo número de disparos, pero generar ocasiones de calidad muy diferente. Del mismo modo, un delantero puede marcar muchos goles en un periodo concreto sin que necesariamente haya generado ocasiones de gran calidad.

El proyecto busca resolver esta limitación mediante una herramienta que mida la calidad de las ocasiones y permita analizar el rendimiento más allá de los resultados.

Actualmente, los aficionados y profesionales consultan estadísticas tradicionales y plataformas de analítica deportiva para estudiar el rendimiento de equipos y jugadores. No obstante, el número de goles y disparos por sí solo no permite conocer la calidad real de las ocasiones generadas.

Los usuarios iniciales serían analistas deportivos, entrenadores, clubes de fútbol, periodistas deportivos y aficionados interesados en profundizar en las estadísticas de equipos y jugadores.

## Uso y valor

El usuario podría consultar las estadísticas de una competición, equipo o jugador y analizar la calidad de las ocasiones generadas y recibidas.

La plataforma calcularía el xG de cada disparo y permitiría:

- Comparar los goles reales con los goles esperados.
- Analizar qué equipos generan ocasiones de mayor calidad.
- Estudiar la eficacia de los delanteros en relación con las oportunidades que tienen.
- Identificar las variables que más influyen en la probabilidad de marcar.
- Visualizar la evolución del rendimiento ofensivo a lo largo de los partidos o temporadas.

El valor principal sería transformar los datos de eventos de fútbol en información útil para comprender mejor el rendimiento deportivo y apoyar la toma de decisiones.

## Datos y fuentes candidatas

- **Datos principales — StatsBomb Open Data:** repositorio público con información de eventos de partidos de fútbol, incluyendo disparos, coordenadas, características de las acciones y resultados.

  [https://github.com/statsbomb/open-data](https://github.com/statsbomb/open-data)

- **Catálogo de competiciones y temporadas:** permitirá identificar qué ligas, temporadas y partidos están disponibles antes de seleccionar la muestra.

  [https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json](https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json)

- **Documentación de StatsBomb:** información sobre la estructura de los archivos JSON, los tipos de eventos y las variables disponibles.

  [https://github.com/statsbomb/open-data/tree/master/doc](https://github.com/statsbomb/open-data/tree/master/doc)

- **Metrica Sports Sample Data (fuente complementaria):** conjunto de datos de muestra con eventos y tracking de fútbol que podría utilizarse para ampliar el análisis visual de determinadas jugadas.

  [https://github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data)

**Estado de acceso:** StatsBomb Open Data es una fuente pública y descargable, por lo que constituye una base viable para comenzar el proyecto. No obstante, será necesario auditar las competiciones, temporadas y variables disponibles antes de determinar el alcance definitivo. No se garantiza una cobertura completa de todas las ligas ni temporadas.

**Privacidad y restricciones:** se trabajará principalmente con datos deportivos públicos. Se respetarán las condiciones de uso y atribución de StatsBomb, evitando redistribuir los archivos originales si sus condiciones no lo permiten. No se utilizarán datos personales privados ni información que no esté disponible en el momento del disparo para realizar las predicciones.

## Primera versión o MVP

1. Descarga e inventario de los datos disponibles de StatsBomb.
2. Procesamiento de los archivos JSON y construcción de una tabla analítica en la que cada fila represente un disparo.
3. Análisis exploratorio de las ocasiones y de variables como distancia, ángulo, parte del cuerpo y contexto de la jugada.
4. Desarrollo de un modelo inicial de regresión logística para estimar la probabilidad de gol.
5. Comparación con modelos de Machine Learning como Random Forest y Gradient Boosting.
6. Evaluación de los modelos mediante métricas predictivas y de calibración.
7. Desarrollo de un dashboard interactivo para consultar xG, goles reales y estadísticas de equipos y jugadores.

**Fuera del MVP:** predicción de resultados de partidos, recomendaciones de apuestas, predicción de lesiones, valoración económica de futbolistas y reconstrucción táctica completa.

## Viabilidad y dudas

**Principal dificultad:** la disponibilidad y calidad de los datos puede variar entre competiciones y temporadas. Además, será necesario evitar fugas de información para que el modelo no utilice variables que revelen el resultado del disparo. También habrá que comprobar que las probabilidades estimadas estén bien calibradas y sean interpretables.

**Qué comprobar:** auditar inicialmente el repositorio de StatsBomb para conocer el número de partidos y disparos disponibles, la cobertura de las variables y los posibles valores ausentes. Se deberá comprobar si existe un volumen suficiente de datos para entrenar y evaluar los modelos, planteando como referencia inicial un mínimo de 10.000 disparos válidos. También se compararán los resultados de distintos modelos y se estudiará la utilidad del dashboard para analizar equipos y jugadores.

**Valor diferencial:** el proyecto no se limitaría a mostrar estadísticas de fútbol, sino que desarrollaría un modelo propio de xG, interpretable y evaluado, que permita estimar la calidad de las ocasiones y obtener conclusiones sobre el rendimiento ofensivo.


# Idea 3. SalaryScope AI — Estimación inteligente del salario bruto anual de desarrolladores

## Título y descripción

SalaryScope AI es una plataforma web que utiliza técnicas de Machine Learning para estimar el salario bruto anualizado de un profesional del desarrollo de software a partir de su perfil: edad, años de experiencia, país, nivel educativo, rol, tecnologías y herramientas que domina.

El usuario introducirá su perfil y el sistema devolverá una estimación salarial con un rango de confianza, junto con los factores que más influyen en el resultado.

El proyecto combinará armonización de datos de varias ediciones de encuesta, análisis exploratorio, entrenamiento y evaluación de modelos predictivos y desarrollo de una interfaz web interactiva.

## Problema y usuario

Saber cuánto se puede cobrar en el mercado laboral es difícil. El salario depende de la experiencia, el país, el rol y las tecnologías, y las fuentes disponibles suelen ofrecer medias generales que no se ajustan a un perfil concreto ni explican qué variables pesan más.

SalaryScope AI pretende ofrecer una referencia orientativa y personalizada, basada en miles de respuestas reales de desarrolladores de todo el mundo.

Actualmente, los desarrolladores suelen consultar encuestas salariales, ofertas de empleo y plataformas que publican rangos salariales para conocer las remuneraciones del sector. Sin embargo, estas fuentes no siempre permiten obtener una estimación adaptada a las características concretas de cada perfil profesional.

Los usuarios iniciales serían estudiantes y recién graduados que quieren conocer su posible salario, profesionales que evalúan una oferta o un cambio de empleo, y reclutadores que necesitan una referencia complementaria.

## Uso y valor

Una persona con 3 años de experiencia, que trabaja con Python y SQL en España, introduce su perfil. La plataforma devuelve un salario bruto anual estimado con un intervalo, y muestra qué variables han pesado más (por ejemplo, país y experiencia) y cómo cambiaría la cifra si añadiera otra tecnología o más años de experiencia.

El valor principal es convertir datos de encuestas en una estimación accesible y comprensible. La estimación se presentaría como una referencia orientativa, no como una oferta salarial ni una garantía.

## Datos y fuentes candidatas

- **Stack Overflow Developer Survey (2012-2025):** [https://survey.stackoverflow.co/](https://survey.stackoverflow.co/) y  (https://github.com/StackExchange/Survey/blob/main/packages/archive/2025/results.csv)

  Encuesta anual a desarrolladores de todo el mundo. Cada edición se descarga como CSV junto con su esquema de columnas. Contiene edad, años de experiencia, país, formación, rol, tecnologías utilizadas y salario bruto anualizado. Licencia ODbL.

- **Archivos de trabajo:** un CSV por año (2012 a 2025) y su archivo de esquema (`survey_results_schema.csv`). Se unificarán en un único conjunto de datos con una columna de año.

  Las columnas cambian entre ediciones (por ejemplo, el salario aparece con nombres distintos según el año y no todas las preguntas existen en todos los años), por lo que será necesario construir un diccionario de equivalencias.

- **Documentación de la fuente:** esquema y metodología de cada edición, para comprobar el significado de cada variable, la moneda, el periodo del salario y las condiciones de uso.

**Estado de acceso:** los datasets de 2012 a 2025 son de descarga pública. Falta realizar la descarga completa, el inventario de columnas por año y la revisión de valores ausentes en el salario.

**Privacidad y restricciones:** las respuestas están anonimizadas. Se respetará la licencia ODbL (atribución y compartir en las mismas condiciones), se documentarán las transformaciones y no se intentará reidentificar a ningún participante.

## Primera versión o MVP

1. **Armonización de datos:** descargar las 14 ediciones, mapear columnas equivalentes y unificar el salario en una única variable (USD anuales).

2. **Auditoría y limpieza:** eliminar registros sin salario, duplicados y valores inconsistentes; tratar valores extremos (por ejemplo, recorte por percentiles).

3. **Análisis exploratorio:** estudiar la distribución del salario (muy asimétrica) y su relación con experiencia, edad, país, formación y rol.

4. **Preparación de variables:** codificar las variables categóricas, convertir las tecnologías (respuestas múltiples) en variables binarias, gestionar valores ausentes y estudiar la transformación logarítmica del salario.

5. **Desarrollo de modelos:** modelo base de regresión y comparación con Random Forest y Gradient Boosting.

6. **Evaluación:** MAE y RMSE (y error relativo, por la asimetría del salario), con separación temporal: entrenar con ediciones anteriores y evaluar con las más recientes (2024-2025).

7. **Interpretabilidad:** importancia de variables, análisis SHAP y estimación de intervalos de predicción.

8. **Aplicación web:** interfaz en Streamlit para introducir el perfil y obtener la estimación.

9. **Informe final:** datos, limpieza, modelos comparados, resultados, limitaciones y mejoras.

## Fuera del MVP

Quedarían fuera de la primera versión las soft skills medidas de forma directa, los perfiles no tecnológicos, la predicción de salarios futuros, las recomendaciones de negociación salarial y la integración con portales de empleo en tiempo real.

Tampoco se presentará como salario de mercado actual una estimación basada en ediciones antiguas sin ajustar ni evaluar su capacidad de representar el presente.

## Viabilidad y dudas

**Principal dificultad:** los salarios de 2012 a 2025 no son directamente comparables. Cambian la moneda, la inflación, la composición de países y las preguntas de cada edición. Habrá que decidir si se ajusta por inflación, se incluye el año como variable o se da más peso a las ediciones recientes.

**Limitación de cobertura:** la encuesta solo recoge desarrolladores y soft skills casi no aparecen como variables. El modelo predecirá a partir de hard skills, experiencia y contexto, y las soft skills quedarían como una ampliación futura con otra fuente de datos.

**Sesgo de muestra:** los participantes se autoseleccionan y el salario es autodeclarado, por lo que la muestra puede sobrerrepresentar ciertos países y perfiles. Los resultados no serán extrapolables a todo el mercado laboral, y habrá que revisar cuántas respuestas hay por país para no ofrecer predicciones poco fiables en zonas con pocos datos.

**Qué comprobar:** el significado y la moneda de la variable salario en cada año, el porcentaje de respuestas con salario, la distribución por países, los valores extremos y si el error del modelo es estable entre regiones y niveles de experiencia.

**Valor diferencial:** SalaryScope AI desarrollará un modelo predictivo propio, entrenado con más de una década de datos armonizados, que ofrece estimaciones individuales con intervalo de confianza y explicación de resultados. Combina Machine Learning aplicado, ingeniería de datos, evaluación rigurosa y desarrollo de producto.

---

# Comparación inicial

## 1. Resumen de las propuestas

| Criterio | HousingPulse AI | GoalMetrics AI | SalaryScope AI |
|---|---|---|---|
| **Ámbito** | Mercado inmobiliario | Analítica deportiva | Mercado laboral tecnológico |
| **Objetivo** | Estimar el precio de una vivienda | Calcular la probabilidad de gol de cada disparo (xG) | Estimar el salario bruto anual de un desarrollador |
| **Fuente principal** | RealEstateDataset | StatsBomb Open Data | Stack Overflow Developer Survey |
| **Disponibilidad de datos** | <code>Dataset descargado</code> | <code>Fuente pública</code> | <code>Descarga pendiente</code> |
| **Usuario objetivo** | Particulares, compradores y profesionales inmobiliarios | Analistas, entrenadores, clubes, periodistas y aficionados | Estudiantes, desarrolladores, profesionales y reclutadores |
| **Aplicación final** | Aplicación web con Streamlit | Dashboard interactivo | Aplicación web con Streamlit |

## 2. Comparación técnica y de viabilidad

| Criterio | HousingPulse AI | GoalMetrics AI | SalaryScope AI |
|---|---|---|---|
| **Datos iniciales** | Aproximadamente un millón de registros de propiedades de Miami-Dade, con características, ubicación, precios y fechas de venta. | Eventos de partidos, disparos, coordenadas y características de las jugadas. | 14 ediciones anuales (2012-2025) con información sobre experiencia, país, formación, rol, tecnologías y salarios declarados. |
| **Estado de acceso** | CSV descargado y primera revisión de estructura y valores ausentes. | Fuente pública y descargable. Pendiente inventariar competiciones, temporadas y disparos. | Datasets públicos. Pendiente descargar las ediciones y armonizar sus columnas. |
| **Riesgo principal** | Calidad y actualidad de los precios, fiabilidad de las fechas y representatividad geográfica. | Cobertura desigual, calidad de los eventos y posibles fugas de información. | Diferencias entre ediciones, cambios de moneda e inflación y sesgos de la muestra. |
| **MVP** | Modelo de estimación de precios y aplicación web para introducir las características de una vivienda. | Modelo propio de xG y dashboard para analizar disparos, equipos y jugadores. | Modelo de estimación salarial y aplicación web para introducir un perfil profesional. |
| **Potencial de ML** | **Alto.** Regresión, Random Forest y Gradient Boosting. | **Alto.** Regresión logística, Random Forest y Gradient Boosting. | **Alto.** Regresión, Random Forest y Gradient Boosting. |
| **Potencial de GenAI** | Explicación de las variables que influyen en el precio y generación de informes. | Explicación de estadísticas y rendimiento de equipos y jugadores. | Explicación de los factores que influyen en el salario y generación de resúmenes. |
| **Incertidumbre clave** | Capacidad del modelo para representar el mercado inmobiliario reciente. | Volumen de datos y generalización entre competiciones y temporadas. | Comparabilidad de salarios entre años y representatividad por países. |

## 3. Valoración inicial de cada propuesta

### HousingPulse AI
- **Punto fuerte:** dispone de un conjunto de datos inicial ya descargado, con un volumen aproximado de un millón de registros.
- **Principal reto:** comprobar la calidad y actualidad de los precios y las fechas de venta.
- **Potencial:** desarrollar un modelo predictivo propio y una aplicación web funcional.

### GoalMetrics AI
- **Punto fuerte:** permite trabajar con datos deportivos de eventos y desarrollar un modelo de Machine Learning con una aplicación práctica en fútbol.
- **Principal reto:** comprobar la cobertura de competiciones y temporadas y evitar fugas de información.
- **Potencial:** construir un modelo xG propio y un dashboard de análisis de rendimiento.

### SalaryScope AI
- **Punto fuerte:** cuenta con varias ediciones de una encuesta pública que permiten estudiar la evolución de los salarios y las características profesionales.
- **Principal reto:** armonizar los datos de distintos años y controlar las diferencias de moneda, inflación y representatividad.
- **Potencial:** desarrollar un modelo de predicción salarial personalizado y explicar los factores que influyen en el resultado.

**Conclusión provisional:** las tres propuestas presentan potencial para desarrollar un producto basado en datos y Machine Learning. La elección definitiva dependerá de la calidad y disponibilidad de los datos, la viabilidad del MVP y la utilidad real para los usuarios.
