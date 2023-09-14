# Inmersion_Datos_Alura

## Índice
* [Introducción](#introducción)
    * [¿De qué trata la base de datos?](#de-qué-trata-la-base-de-datos)
* [Aura 01](#aura-01-tu-primer-colab-con-python-y-pandas)
    * [Desafío 1](#desafío-1)
    * [Desafío 2](#desafío-2)
* [Aula 02](#aula-02-tratamiento-de-datos-y-primeras-gráficas)
    * [Desafío 1](#desafío-1-1)
    * [Desafío 2](#desafío-2-1)
* [Aula 03](#aula-03-profundizando-el-análisis-exploratorio-e-insights)
    * [Desafío 1](#desafío-1-2)
    * [Desafío 2](#desafío-2-2)

## Introducción

La Inmersión en Datos de Alura es un evento que organiza [Alura Latam](https://www.aluracursos.com/) como entrenamiento para Data Science. Durante 5 días se liberarán clases diarias que explicarán de forma intensiva como realizar análisis de datos con Python usando [Pandas](https://pandas.pydata.org/) y otros frameworks, como [Matplotlib](https://matplotlib.org/stable/index.html) y [Seaborn](https://seaborn.pydata.org/), para la visualización. Todo el proceso del la Inmersión se registrará en el archivo [Inmersion_de_datos.ipynb](Inmersion_de_datos.ipynb).

La Inmersión en Datos también consta de varios desafíos diarios para los participantes, y en este README llevaré a cabo la resolución de estos. Cada días se libera un Aura numerada del 1 al 5, y aquí se mostrarán las resoluciones de sendos desafíos. En mi caso, La Inmersión en Datos durará desde el 12 al 15 de septiembre del 2023.

### ¿De qué trata la base de datos?

La base de datos está dada por el archivo CSV [inmuebles_bogota.csv](inmuebles_bogota.csv) que resume los inmuebles en venta de la ciudad de Bogotá. El dataframe es el siguiente

```python
import pandas as pd

inmuebles = pd.read_csv('./inmuebles_bogota.csv')
inmuebles.head()
```
![imagen](https://github.com/leodan52/Inmersion_Datos_Alura/assets/109176490/fee275e9-2ce6-4db4-a765-870e91737213)

Como se puede observar, el dataframe compila varios atributos de los inmuebles en venta, como el tipo, la cantidad de habitaciones, la ubicación, etcétera. Como se puede observar, los datos están en español, lo que podría dar problemas al usar python cuando tengamos que tratar con vocales con acentos y la letra ñ. Por eso es necesario renombrar esas columnas problemáticas.


```python
#Creamos un diccionario con los reemplazos para las columnas a renombrar
columnas = {
    'Baños' : 'Banios',
    'Área' : 'Area'
}
# Renombramos las columnas
inmuebles = inmuebles.rename(columns = columnas)
inmuebles.sample(10)
```
![imagen](https://github.com/leodan52/Inmersion_Datos_Alura/assets/109176490/8d620a55-cad5-4fbd-8c52-2fcbfd7bb4fa)

y de esta forma será más sencillo trabajar.

## Aura 01: Tu primer colab con python y pandas
**Desafíos:**
1. Promedio de área de todos los barrios, y realizar un gráfico con el top 10
2. Consultar otros datos estadísticos: conteo, mediana, valores mínimos y máximos.

### Desafío 1

Para este desafío me ayudé del método de pandas [`groupby`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html), que permite agrupar por un campo los registros de la base de datos; es [muy parecido a la función que se usa en MySQL](https://pandas.pydata.org/docs/getting_started/comparison/comparison_with_sql.html). De esta forma será más sencillo agrupar los diferentes barrios que aparecen en el dataframe, y calcular sendos promedios de la áreas de los inmuebles.

```python
area_estadistica = inmuebles[['Barrio','Area']].groupby('Barrio').describe()
area_estadistica.columns = [dato for _,dato in area_estadistica.columns]

area_estadistica
```
![imagen](https://github.com/leodan52/Inmersion_Datos_Alura/assets/109176490/350a0b24-4b09-4c52-8a1e-4bd49fa401fb)

Nótese que también se usó el método [`describe`](https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.describe.html#pandas.core.groupby.DataFrameGroupBy.describe) que no solo calcula el promedio (mean), sino también varios parámetros estadísticos que no ayudarán con el siguiente desafío.

Ya teniendo el dataframe `area_estadistica` realizar el gráfico es bastante sencillo. Ya que la cantidad de datos es bastante grande, vamos a graficar solo el Top 10 de Barrios con un promedio de área por inmueble mayor.

```python
area_promedio = area_estadistica['mean'].sort_values(ascending = False)

area_promedio.head(10).plot(kind = 'bar', title = 'Top 10 Promedio de Áreas', ylabel = 'Área (m^2)')
```
![imagen](https://github.com/leodan52/Inmersion_Datos_Alura/assets/109176490/562eef62-f300-42e4-8bcf-789a75269e9c)

#### Análisis

Como podemos observar, la diferencia del top 1 al top 2 es bastante marcada; siendo con Modelia alrededor de $4500 m^2$ y Libertador apenas de alrededor de $800 m^2$. Esto nos lleva a la pregunta, ¿qué clase de inmuebles hay en Modelia para tener tanta extensión? Verifiquemos eso.

```python
inmuebles[inmuebles.Barrio == 'Modelia'].Tipo.value_counts()
```
![imagen](https://github.com/leodan52/Inmersion_Datos_Alura/assets/109176490/e82a2b23-948d-4533-a2be-7bcca782a2e9)

Podemos observar que en Modelia solo tenemos casas y departamentos en cantidades similares. Promediando esto dimensiones tan grandes, quizás Modelia se trate de una zona residencial con costos de vida bastante altos, aunque también hay que revisar otros parámetros estadísticos, como la desviación estándar, para verificar esa afirmación.
```python
area_estadistica.loc['Modelia']
```
![imagen](https://github.com/leodan52/Inmersion_Datos_Alura/assets/109176490/6f3abc4c-5302-42dd-b49f-bbc02a42694e)

Al parecer los datos están muy dispersos, dada la desviación estándar. El tercer cuartil tiene un valor bastante pequeño, por lo que los datos grandes son bastante escasos. Siendo los datos más grandes tan escasos, se podría de tratar de errores en la captura de datos, o bien, un barrio un tanto diverso.

### Desafío 2

Para este desafío se usará de nuevo el dataframe `area_estadistica` ya que posee los parámetros estadísticos del área que usaremos para este desafió. Hemos renombrado los campos para tener nombres más entendibles para hispanohablantes.

```python
columnas = {
    'count' : 'contador',
    'mean' : 'promedio',
    'std' : 'd_estandar',
    '25%' : 'qt1',
    '50%' : 'mediana',
    '75%' : 'qt3'
}

area_estadistica = area_estadistica.rename(columns = columnas)

area_estadistica
```
![imagen](https://github.com/leodan52/Inmersion_Datos_Alura/assets/109176490/9be3870b-d18b-4f12-a827-b80db31e827f)

Esta tabla renombrada ya se había usado anteriormente.

Ahora vamos a mostrar una serie de gráficas Top 10 para algunos de los parámetros estadísticos.

#### Gráficos

**Desviación estándar**

```python
area_d = area_estadistica['d_estandar'].sort_values(ascending = False)

area_d.head(10).plot(kind = 'bar', title = 'Top 10 desviación estándar del total de áreas', ylabel = 'Área (m^2)')
```
![imagen](https://github.com/leodan52/Inmersion_Datos_Alura/assets/109176490/df33a423-b7fd-4160-8c3d-b9373664fb23)

**Contador (Histograma)**

```python
area_contador = area_estadistica['contador'].sort_values(ascending = False)

area_contador.head(10).plot(kind = 'bar', title = 'Top 10 barrios con más inmuebles en venta', ylabel = 'count')
```
![imagen](https://github.com/leodan52/Inmersion_Datos_Alura/assets/109176490/dd21b313-279c-4997-9840-6fb50e0e5f26)

**Valor Mínimo**

```python
area_min = area_estadistica['min'].sort_values(ascending = False)

area_min.head(10).plot(kind = 'bar', title = 'Top 10 valores mínimos por barrio', ylabel = 'Área (m^2)')
```
![imagen](https://github.com/leodan52/Inmersion_Datos_Alura/assets/109176490/6dea44a4-3b4b-4c7b-9984-1e3a61160a46)

**Valor Máximo**

```python
area_max = area_estadistica['max'].sort_values(ascending = False)

area_max.head(10).plot(kind = 'bar', title = 'Top 10 valores máximos por barrio', ylabel = 'Área (m^2)')
```
![imagen](https://github.com/leodan52/Inmersion_Datos_Alura/assets/109176490/34db1363-948b-4cf7-9b13-cfe470653bff)

**Mediana**

```python
area_mediana = area_estadistica['mediana'].sort_values(ascending = False)

area_mediana.head(10).plot(kind = 'bar', title = 'Top 10 mediana de áreas por barrio', ylabel = 'Área (m^2)')
```

![imagen](https://github.com/leodan52/Inmersion_Datos_Alura/assets/109176490/87e1294b-2167-45fc-a219-60b24a43a812)

## Aula 02: Tratamiento de Datos y Primeras Gráficas

**Desafíos**
1. Estudiar mejor el histograma de valores, seleccionar 3 tipos de inmuebles (refinar el gráfico).
2. Precio del metro cuadrado por barrio, y encontrar un gráfico adecuado para esta nueva variable.

En el aula 02 se trató la columna Valor del dataframe para obtener los valores numéricos de los inmuebles. El dataframe quedó de la siguiente forma

```python
inmuebles.sample(10)
```
![imagen](https://github.com/leodan52/Inmersion_Datos_Alura/assets/109176490/09a42e1e-c0c2-4217-9c50-59a31ad50bcd)

Se agregaron varias columnas, pero lo que nos interesa ahora es la columna 'Precio_Millon' para resolver los desafíos. Este número es el valor, en millones de pesos colombianos (COP) de cada inmueble.

### Desafío 1

El histograma de valores es el siguiente:

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (10,6))

grafica = sns.histplot(data = inmuebles, x = 'Precio_Millon', kde = True, hue = 'Tipo')
grafica.set_title('Distribución de Valores en los inmuebles en Bogotá')

plt.xlim((50,1000))
plt.savefig('./img/valor_unmueble.png', format = 'png')
plt.show()
```
![imagen](img/valor_unmueble.png)

Vamos a elegir el top 3 de tipos de inmuebles que más se repiten.

```python
inmuebles.Tipo.value_counts()
```
![imagen](https://github.com/leodan52/Inmersion_Datos_Alura/assets/109176490/f66d7b7a-600b-43b1-8058-f7f7c09c7b44)

Por lo visto, el top 3 de tipos de inmuebles son Apartamento, Casa y Oficina/Consultorio. Vamos a limitar el histograma anterior a estos tres tipos. Creamos una lista que usaremos para el filtro.

```python
tipo_sel = ['Apartamento', 'Casa', 'Oficina/Consultorio']
```
Luego, creamos el gráfico, modificando algunos detalles visuales para darle una mejor presentación.

```python
plt.figure(figsize = (12,8)) # Creamos lienzo y ajustamos tamanio
sns.set_theme(font_scale = 1.5) # Ajustamos el tamanio de la fuente

grafica = sns.histplot(data = inmuebles_simpl, x = 'Precio_Millon',
                       kde = True,
                       hue = 'Tipo',
                       palette = 'dark',  # Cambiamos colores
                       multiple="stack"   # Evitamos que las barras sean transparentes
                       )

grafica.set_title('Distribución de precios en los inmuebles en Bogotá')
grafica.set_xlabel('Precio (Millones COP)')
grafica.set_ylabel('Frecuencia')

plt.xlim((50,2000))
plt.savefig('./img/valor_inmueble_top3.png', format = 'png')
plt.show()
```

![imagen](img/valor_inmueble_top3.png)

Lo que tenemos en el gráfico es la distribución de los valores de los tres tipos de inmueble entre 50 7 2000 millones COP. Como se puede observar, la cantidad de Apartamentos en oferta domina en general a los otros dos tipos, tanto que es difícil observarlos. Hagamos un par de zoom in para apreciarlos de mejor manera.

```python
plt.figure(figsize = (12,8))
sns.set_theme(font_scale = 1.5)

grafica = sns.histplot(data = inmuebles_simpl, x = 'Precio_Millon', kde = True, hue = 'Tipo', palette = 'dark',  multiple="stack")
grafica.set_title('Distribución de precios en los inmuebles en Bogotá')
grafica.set_xlabel('Precio (Millones COP)')
grafica.set_ylabel('Frecuencia')

plt.xlim((50,2000))
plt.ylim((0,110))
plt.savefig('./img/valor_inmueble_top3_zoomin_1.png', format = 'png')
plt.show()
```
![imagen](img/valor_inmueble_top3_zoomin_1.png)


```python
plt.figure(figsize = (12,8))
sns.set_theme(font_scale = 1.5)

grafica = sns.histplot(data = inmuebles_simpl, x = 'Precio_Millon', kde = True, hue = 'Tipo', palette = 'dark',  multiple="stack")
grafica.set_title('Distribución de precios en los inmuebles en Bogotá')
grafica.set_xlabel('Precio (Millones COP)')
grafica.set_ylabel('Frecuencia')

plt.xlim((50,2000))
plt.ylim((0,7))
plt.savefig('./img/valor_inmueble_top3_zoomin_2.png', format = 'png')
plt.show()
```
![imagen](img/valor_inmueble_top3_zoomin_2.png)

#### Análisis

De estos gráficos se pueden llegar a una serie de conclusiones:
1. La oferta de departamentos de bajo costo es bastante alta. Quizás haya habido algún proyecto inmobiliario destinado a las viviendas multi-familiares.
2. La oferta tanto de departamentos como de casas tiene un comportamiento bastante suave entre 50 y 1000 millones COP, después de eso hay "baches" en algunos precios. Esto podría se indicativo de la capacidad económica de los habitantes de Bogotá.
3. Las Oficinas y consultorios, aunque son el top 3, la diferencia es bastante marcada con el top 2. La grafica no es en absoluto suave, y no presenta un comportamiento predecible. Tiene sentido, ya que es un mercado muy de nicho.

### Desafío 2

Vamos a crear una nueva columna, donde guardaremos el costo por $m^2$ que se está cobrando por cada vivienda. La formula es sencilla
$$costo\ por\ m^2 = \frac{costo\ total\ del\ inmueble}{Área}$$
Por lo que, en código queda de la siguiente forma
```python
inmuebles['Precio_por_m2'] = inmuebles.Precio_Millon / inmuebles.Area

inmuebles.sample(10)
```
![imagen](https://github.com/leodan52/Inmersion_Datos_Alura/assets/109176490/5e32b378-d48f-4fda-a5df-a5bbead4245a)

Donde la nueva columna `Precio_por_1m2` nos dice el costo en millones de COP por cada $m^2$.

#### Grafica

Para la gráfica opté por usar `barplot` para graficar los promedios del valor por $m^2$ de cada barrio. Comenzamos obteniendo los promedios,

```python
datos_est = inmuebles.groupby(by = 'Barrio').describe()

top_promedio_porM2 = datos_est.Precio_por_m2[['mean', 'std']].sort_values(by = 'mean', ascending=False).head(10)
top_promedio_porM2 = top_promedio_porM2.reset_index()
top_promedio_porM2
```
![imagen](https://github.com/leodan52/Inmersion_Datos_Alura/assets/109176490/d7ee932d-c5a0-4282-9f63-dbda864a4734)

además también se obtiene la desviación estándar, con la finalidad de generar **barras de error**.

Creamos la gráfica:
```python
# Reiniciamos settings
sns.set_theme()
plt.figure(figsize = (10,8))

sns.barplot(data = top_promedio_porM2, y = 'mean', x = 'Barrio')

#Obtenemos la desviación estandar
sigma = top_promedio_porM2['std']

x = top_promedio_porM2['Barrio']
y = top_promedio_porM2['mean']

# Graficamos las barras de error
plt.errorbar(x = x, y = y, yerr=(sigma, sigma), fmt='none', color='k', capsize=5)

plt.title('Top 10 Promedio del costo por m2 por barrio')
plt.xlabel('Barrio')
plt.ylabel('Promedio (millones COP)')

# Rotamos las etiquetas de x para que los nombres no se solapen
plt.xticks(rotation = 80)

plt.savefig('./img/top10_primedio_costo_m2.png', format = 'png')
plt.show()
```
![imagen](img/top10_primedio_costo_m2.png)

La altura de las barras llega al promedio de los datos, y las barras de error, que se extienden una desviación por arriba y por abajo, muestras la variación que tienen los datos alrededor del promedio.

#### Análisis

Como podemos observar, el top 10 de precio por $m^2$ difiere mucho del top 10 de costos total de inmueble. En este caso el barrio El Virrey se lleva el top 1, además que los demás no se desvían demasiado. La desviación estándar también es bastante aceptable, siendo menor a 3 para todos los casos, lo que no dice que los precios son bastante consistentes en cada barrio.

Es bastante notorio el caso de La Merced, que ocupa el top 3, donde también tiene la desviación estándar menor. Esto indica que los costos por $m^2$ en ese barrio son bastante estables.

## Aula 03: Profundizando el análisis exploratorio e insights
**Desafíos**

1. Dar un vistazo a la base de datos del DANE, entender estas variables, conceptualmente para entender mejor el contexto de esta base.
2. Organizar tu notebook para tu proyecto quede mejor presentado

### Desafío 1

En esta ocasión los desafíos son más teóricos que las clases pasadas, por lo tanto el código que se puede escribir será mínimo. Por lo tanto, a modo de solución, escribiré en esta sección un resumen de la investigación de los archivos CSV que usaremos en la siguiente clase.

Los datos del DANE se obtuvieron de su [página web](https://microdatos.dane.gov.co/index.php/catalog/743), y los archivos CSV descargados el 13 de septiembre del 2023 se encuentran en el directorio [datos_varios_colombia/](datos_varios_colombia/). La investigación de estos archivos se realiza con la [documentación](datos_varios_colombia/ddi-documentation-spanish-743.pdf) que dispone también el DANE.

#### Identificación (Capítulo A)

Esta archivo se guardó en la variable `datos_raw` durante la clase, para después hacer un filtro que permitiera quedarnos solo con los datos de Bogotá —municipio con indicador 11001. Esta archivo **contiene datos de identificación de la vivienda**, como Departamento, municipio, clase, etcétera; en total posee 11 columnas.

Las columnas, y su abreviatura usada en el archivo, se resume en la siguiente tabla

![f1.3](docs/assets/f1-3.png)

En la columna **NOMBRE** se muestra el nombre que tendrá la columna o campo en el archivo, en **ETIQUETA** la descripción y formato en el que se mostrará la información, y en **PREGUNTA** la pregunta que se realizó al encuestado.

Esta tabla es la básica al tratar con un modelo predictivo, ya que nos brinda información de la zona y el tipo del inmueble al que se refiere, lo que es crucial a la hora de asignar un valor a la propiedad.

#### Datos de la vivienda y su entorno (Capítulo B)

En este caso, se creó un dataframe con este archivo llamado `datos_b` durante la clase. Esta contiene **los datos propios de la vivienda, así como de su entorno**. Por ejemplo, las vías de acceso, iluminación, condiciones estructurales, posibles problemas ambientales, etcétera. Este archivo posee 63 campos.

La explicación del los datos se expone en el siguiente fragmento

![f2.3](docs/assets/f2-3.png)

Los campos constan del **DIRECTORIO** así como de varios más cuyo nombre clave comienza con **NVCBP** y el factor de expansión (**FEX_X**). Las columnas NVCBP constan de la información dada por los encuestados sobre el estado de su viviendo y su entorno, como se muestra en la tabla. Los datos de estos campos son numéricos discretos, comenzando desde 1, enumerando dos o varias respuestas posibles a las preguntas.

Un posible abordaje de los datos mostrados en este dataset sería generalizar los datos a zonas más amplias como las UPZ o las localidades, haciendo uso del archivo anterior, y dar con un estado general de las viviendas en vecindades.

#### Condiciones habitacionales del hogar (Capítulo C)

Este archivo se guardó en la variable `datos_c`. Posee información del **estado administrativo de la vivienda**. Por ejemplo, si es propia o arrendada, los gastos por habitar la vivienda, la financiación de los hogares para comprar la vivienda, la calidad de los espacios, etcétera. Este archivo consta de 142 columnas.

![f3.3](docs/assets/f3-3.png)

En este caso, la tabla consta de datos de identificación, y los resultados de la encuentras con campos que comienzan con **NHCCP**. Estos datos son numéricos, contando con datos discretos con un conjunto de respuestas disponibles, así como con continuos para respuestas más específicas que involucren cantidades.

En este dataframe se podría obtener información sobre el costo de vida que la vivienda exige para su mantenimiento, lo que influye en su costo de venta.

#### Composición del hogar y demografía (Capítulo E)

Guardado en la variable `datos_e`, este dataset habla de la **composición del hogar y la demografía**. Datos como el tamaño, distribución y dinámica demográfica de la población, sus características demográficas, integrantes del hogar y el núcleo familiar, etcétera. Este archivo consta con 68 campos.

![f4.3](docs/assets/f4-3.png)

Además de los datos de identificación, tenemos también los datos que se obtienen de la encuesta, en este caso comenzando por **NPCEP**, así como un par de datos que se salen de este patrón como **ORDEN** y **SEXO** del encuestado. Por lo demás, el formato sigue siendo dado por datos numéricos discretos y continuos, para respuestas definidas o específicas, respectivamente.

Los datos contenidos en este archivo poseen datos referentes más a los residentes del hogar, lejos de la vivienda. Quizás sean útiles para entender la dinámica de los posibles vecinos al adquirir un hogar, lo que impacta al mercado inmobiliario de una forma u otra.


#### Educación (Capítulo H)

Se guarda en la variable llamada `datos_h`. Como su nombre lo indica **recopila datos educativos de la población**, desde los 5 años en adelante. Parámetros como el analfabetismo, niveles educativos y años de estudio, así como las razones del ausentismo escolar y los datos de transporte y económicos relacionados con la asistencia a los centros de estudio. Este archivo consta de 136 campos.

![f5.3](docs/assets/f5-3.png)

Los tipos de campos no difieren mucho de los archivos anteriores. Se cuenta con los campos de identificación, así como los resultados de la encuesta, esta vez identificada con el prefijo **NPCHP** con datos numéricos discretos para un número fijo de respuestas posibles, y datos numéricos continuos para respuestas que requieran cantidades específicas.

No es extraño que el ambiente escolar sea un tema importante al adquirir una vivienda, especialmente a hogares que cuentan con menores de edad. La accesibilidad a los centro educativos, así como la calidad de estos influyen en los costos de los inmuebles, ya sean residenciales o los locales para negocios que dependan de la actividad educativa, como la venta de útiles escolares.


#### Fuerza de trabajo (Capítulo K)

Guardado en la variable `datos_k`, esta contiene **los datos laborales y de actividad económica de la población**. Recopila datos como el origen de los ingresos de las personas dentro del hogar, ya sean adquiridos mediante el mercado laboral o de otras fuentes, así los datos de desempleo. Este archivo cuenta con 170 campos.

![f6-3](docs/assets/f6-3.png)

En este caso los campos que contienen las respuestas de la encuesta comienzan con **NPCKP**, y siguen siendo datos numéricos discretos o continuos, así como los campos de identificación.

La información de la fuerza de trabajo es importante, ya que la accesibilidad a los centros de trabajo influye a aquellas personas que deseen adquirir una vivienda, así como a empresas que deseen contratar a empleados que vivan cerca de las instalaciones.

#### Percepción sobre las condiciones de vida y el desempeño institucional (Capítulo L)

Este dataset se guarda en la variable `datos_l`, y guarda datos sobre la **percepción de la población sobre la calidad de vida en los alrededores de su vivienda**. Datos sobre las victimas de actividades delictivas, así como la calidad de las instalaciones urbanas y grupos vulnerables, entre otras cosas, se recopilan en este archivo. Contiene 179 campos.

![f7.3](docs/assets/f7-3.png)

En este caso, los campos que contienen los resultados de la encuesta se identifican porque comienzan con **NHCLP**, y constan de datos numéricos discretos y continuos, aunque en su gran mayoría sean discretos con posibles respuestas definidas.

No cabe duda que el nivel de criminalidad impacta la calidad de vida de las personas en una zona, y la calidad de vida es fundamental cuando hogares o empresas buscan asentarse en un lugar. Por ende, el mercado inmobiliario se verá afectado.

#### Conclusiones de la actividad

Estos archivos son los seleccionados por los instructores de la **Inmersión en Datos de Alura**, que serán útiles en la actividad de la Aula 04 en la que se creará un modelo predictivo para el valor de las propiedades.

No me cabe duda que se realizó un buen trabajo en esta selección, ya que las condiciones del barrio, de la vivienda, el nivel educativo de la zona, las oportunidades de empleo y la percepción de la criminalidad, entre otras, son variables muy importantes para valuar un inmueble de manera más precisa.

Debe haber un equilibrio entre el atractivo de la vivienda, teniendo en cuenta el entorno y los servicios estatales ofrecidos, y el impacto económico de esta. Por ejemplo, si el barrio no es muy atractivo, el precio podría ser suficientemente bajo para compensarlo y cerrar la compra.

### Desafío 2

A lo largo de los días anteriores, me he esforzado para mantener organizado mi [notebook](Inmersion_de_datos.ipynb) dividiendo en secciones y subsecciones para encontrar de forma más rápida y cómoda los detalles que se requieran repasar. Además, he puesto bloques de texto explicando lo que se está realizando en ese momento.

El contenido de las cuatro aulas está contenida dentro del mismo notebook, y divididas por secciones. Dentro de cada sección se abren subsecciones cuando se introduce un nuevo concepto, se realiza un análisis específico, se crean gráficos y se definen los desafíos.

Elegí esta organización para aprovechar las importaciones y las variables que se han hecho en aulas anteriores. Sin embargo, esto también significa sacrificar la eficiencia de los códigos. Después de alrededor de 382 bloques de código, se dejan variables que ya no se usan y ocupan espacio en la memoria virtual.

Por eso, para este desafío voy a agregar una subsección al final de cada sección donde se eliminarán variables que ya no se usarán en el futuro. Para lograr esto usaré la función `dir()` de Python sin argumentos, que devuelve una lista que contiene los nombres de las variables dentro de scope global.

```python
dir()
```
![f8.3](docs/assets/f8-3.png)

De esta forma podré localizar de forma más rápida aquellas variables que no usaré después, y borrarlas usando `del`.
