\# WeatherAUS Rain Prediction  

Predicción de RainTomorrow en Australia usando estadística multivariada



Python 3.9+ | Pandas | Scikit-learn | Statsmodels | PCA | LDA | QDA



---

\# WeatherAUS Rain Prediction  

Predicción de RainTomorrow en Australia usando estadística multivariada



Python 3.9+ | Pandas | Scikit-learn | Statsmodels | PCA | LDA | QDA



---



\## Índice

1\. Overview  

2\. Contexto climático de Australia y márgenes del clima  

3\. Dataset: WeatherAUS  

4\. Análisis exploratorio de datos (EDA)  

5\. Metodología  

6\. Resultados  

7\. Análisis de márgenes climáticos  

8\. Instalación y reproducibilidad  

9\. Limitaciones  

10\. Trabajo futuro  

11\. Referencias y licencia  



---



\## 1. Overview



Este proyecto desarrolla un análisis completo de ciencia de datos aplicado a la predicción de lluvia en Australia, utilizando el dataset WeatherAUS. El objetivo principal es predecir la variable binaria \*\*RainTomorrow\*\* a partir de múltiples variables meteorológicas observadas el día actual.



El enfoque del proyecto es deliberadamente estadístico y multivariado, priorizando interpretabilidad, consistencia física y robustez frente a condiciones climáticas extremas. En lugar de utilizar modelos complejos de caja negra, se emplean técnicas clásicas bien fundamentadas:



\- Análisis exploratorio de datos (EDA)

\- Análisis de Componentes Principales (PCA)

\- Regresión logística

\- Análisis discriminante lineal (LDA)

\- Análisis discriminante cuadrático (QDA)



El dataset contiene aproximadamente 145 000 observaciones provenientes de 49 estaciones meteorológicas australianas, con un marcado desbalance de clases: cerca del 77% de los días no presentan lluvia al día siguiente.



Este proyecto demuestra cómo la estadística multivariada permite capturar la estructura atmosférica subyacente del clima australiano y evaluar, de forma crítica, los límites de predicción en regiones áridas y de transición climática.



---



\## 2. Contexto climático de Australia y márgenes del clima



Australia constituye un caso de estudio excepcional para análisis climáticos debido a su diversidad de regímenes atmosféricos. En un solo continente coexisten climas tropicales, subtropicales, templados oceánicos y desérticos, lo que genera fuertes contrastes espaciales y temporales.



Desde el punto de vista de la clasificación de Köppen–Geiger (actualización 2021), los principales climas representados en el dataset son:



\- Cfa: subtropical húmedo, con lluvias distribuidas durante el año y temperaturas invernales superiores a 10 °C (ej. Sydney, Brisbane).

\- Aw: clima tropical de sabana, con estación seca marcada y precipitaciones anuales menores a 750 mm (ej. Darwin).

\- BSh/BWh: estepas y desiertos cálidos, caracterizados por una relación precipitación/evapotranspiración potencial menor a 0.5 (ej. Alice Springs).

\- Cfb: templado oceánico, con temperaturas moderadas y lluvias relativamente regulares (ej. Melbourne, Hobart).



El concepto central de este proyecto es el análisis de los \*\*márgenes climáticos\*\*, entendidos como regiones o condiciones donde el sistema atmosférico opera cerca de umbrales críticos. En estos márgenes, pequeñas variaciones en presión, humedad o temperatura pueden desencadenar cambios abruptos en la ocurrencia de lluvia.



A partir del análisis empírico del dataset WeatherAUS, se identifican los siguientes umbrales relevantes:



\- Rainfall:

&nbsp; - Percentil 5 ≈ 0 mm (condiciones secas)

&nbsp; - Percentil 95 ≈ 50 mm (eventos extremos)

\- MinTemp:

&nbsp; - Valores bajo 0 °C asociados a heladas y atmósfera estable

\- MaxTemp:

&nbsp; - Valores superiores a 40 °C asociados a olas de calor y baja probabilidad de lluvia

\- Pressure:

&nbsp; - Presiones menores a 1008 hPa asociadas a frentes lluviosos

\- Humidity3pm:

&nbsp; - Valores inferiores a 30% asociados a riesgo de incendios

&nbsp; - Valores superiores a 80% asociados a alta probabilidad de precipitación



Uno de los resultados más importantes del proyecto es que los modelos multivariados presentan mejor desempeño en climas húmedos o de transición (Cfa, Cfb) y un desempeño significativamente menor en climas áridos (BWh, BSh). Esto es consistente con la literatura climática y con el informe IPCC AR6, que destaca la dificultad de predecir precipitación en regiones dominadas por variabilidad episódica y forzantes de gran escala como ENSO.



---



\## 3. Dataset: WeatherAUS



Fuente principal: UCI Machine Learning Repository y Bureau of Meteorology (Australia).



Características generales:

\- Observaciones: aproximadamente 145 000

\- Estaciones meteorológicas: 49

\- Periodo temporal: 2013–2016

\- Variable objetivo: RainTomorrow (Yes / No)



Principales columnas del dataset:



| Variable | Tipo | % Missing | Descripción |

|--------|------|-----------|-------------|

| Date | Fecha | 0% | Fecha de observación |

| Location | Categórica | 0% | Estación meteorológica |

| MinTemp | Numérica | ~1% | Temperatura mínima |

| MaxTemp | Numérica | ~1% | Temperatura máxima |

| Rainfall | Numérica | ~2% | Precipitación diaria |

| Evaporation | Numérica | ~43% | Evaporación |

| Sunshine | Numérica | ~48% | Horas de sol |

| Humidity9am | Numérica | ~2% | Humedad relativa a las 9am |

| Humidity3pm | Numérica | ~3% | Humedad relativa a las 3pm |

| Pressure9am | Numérica | ~10% | Presión a las 9am |

| Pressure3pm | Numérica | ~10% | Presión a las 3pm |

| RainToday | Binaria | ~2% | Llovió hoy |

| RainTomorrow | Binaria | ~2% | Lloverá mañana |



Las variables Evaporation y Sunshine presentan un porcentaje elevado de valores faltantes, lo que condiciona su uso en modelos multivariados y obliga a estrategias de imputación cuidadosas.



---



\## 4. Análisis exploratorio de datos (EDA)



El EDA revela patrones consistentes con la física atmosférica:



\- Rainfall presenta una distribución altamente asimétrica, con una gran concentración de ceros.

\- Temperaturas muestran distribuciones aproximadamente unimodales y correlaciones muy altas entre sí.

\- Presión atmosférica presenta baja varianza pero altísima correlación entre mediciones de la mañana y la tarde.

\- Humedad relativa presenta correlación negativa con la temperatura máxima.

\- Existe un desbalance significativo en la variable objetivo: aproximadamente 77% de observaciones corresponden a “No Rain”.



Estos hallazgos justifican el uso de PCA para reducir redundancia y estabilizar los modelos de clasificación.



---



\## 5. Metodología



El pipeline metodológico del proyecto sigue los siguientes pasos:



1\. Carga y limpieza de datos con pandas.

2\. Análisis exploratorio de distribuciones, correlaciones y valores faltantes.

3\. Imputación de valores faltantes mediante media o KNN, según la variable.

4\. Codificación de variables categóricas (Location mediante one-hot, RainToday como binaria).

5\. Estandarización de variables numéricas.

6\. Aplicación de PCA reteniendo componentes que explican al menos el 85% de la varianza.

7\. Entrenamiento de modelos:

&nbsp;  - Regresión logística

&nbsp;  - LDA

&nbsp;  - QDA

8\. Evaluación mediante validación cruzada de 5 folds y métricas adecuadas para clases desbalanceadas.



El PCA identifica cuatro componentes principales con interpretación física clara:

\- PC1: componente térmico

\- PC2: estabilidad atmosférica (presión vs humedad)

\- PC3: viento frente a humedad

\- PC4: precipitación



---



\## 6. Resultados



Desempeño promedio en validación cruzada:



| Modelo | Accuracy | Recall (Yes) | Precision (Yes) | AUC |

|------|----------|--------------|-----------------|-----|

| Regresión logística + PCA | ~0.84 | ~0.72 | ~0.65 | ~0.85 |

| LDA | ~0.83 | ~0.70 | ~0.63 | ~0.83 |

| QDA | ~0.81 | ~0.68 | ~0.60 | ~0.82 |



La regresión logística combinada con PCA ofrece el mejor balance entre interpretabilidad y desempeño predictivo.



Las cargas del PCA muestran que Humidity3pm y Pressure9am son variables clave para anticipar la lluvia, lo cual concuerda con el conocimiento meteorológico.



---



\## 7. Análisis de márgenes climáticos



El análisis por locación revela que:



\- En estaciones costeras húmedas (Sydney, Brisbane), el modelo logra altos valores de recall.

\- En estaciones áridas del interior (Alice Springs), el modelo tiende a predecir sistemáticamente “No Rain”.

\- Los errores del modelo se concentran en días con Rainfall bajo pero no nulo (<1 mm), que representan transiciones difíciles de clasificar.



Este comportamiento confirma que los márgenes climáticos constituyen el principal límite para modelos estadísticos lineales.



---



\## 8. Instalación y reproducibilidad



Requisitos:

\- Python 3.9+

\- pandas

\- numpy

\- scikit-learn

\- statsmodels

\- matplotlib

\- seaborn

\- jupyter



Ejecución básica:



pip install -r requirements.txt

jupyter notebook ProyectoFinalEstadisticaMultivareada-1.ipynb



yaml

Copiar código



El notebook contiene todas las celdas necesarias para reproducir los resultados.



---



\## 9. Limitaciones



\- No se incorporan coordenadas geográficas ni relaciones espaciales explícitas.

\- La calidad de los datos depende de estaciones individuales.

\- Evaporation y Sunshine presentan missing data estructural.

\- No se incluyen índices climáticos de gran escala como ENSO.



\## Índice

1\. Overview  

2\. Contexto climático de Australia y márgenes del clima  

3\. Dataset: WeatherAUS  

4\. Análisis exploratorio de datos (EDA)  

5\. Metodología  

6\. Resultados  

7\. Análisis de márgenes climáticos  

8\. Instalación y reproducibilidad  

9\. Limitaciones  

10\. Trabajo futuro  

11\. Referencias y licencia  



---



\## 1. Overview



Este proyecto desarrolla un análisis completo de ciencia de datos aplicado a la predicción de lluvia en Australia, utilizando el dataset WeatherAUS. El objetivo principal es predecir la variable binaria \*\*RainTomorrow\*\* a partir de múltiples variables meteorológicas observadas el día actual.



El enfoque del proyecto es deliberadamente estadístico y multivariado, priorizando interpretabilidad, consistencia física y robustez frente a condiciones climáticas extremas. En lugar de utilizar modelos complejos de caja negra, se emplean técnicas clásicas bien fundamentadas:



\- Análisis exploratorio de datos (EDA)

\- Análisis de Componentes Principales (PCA)

\- Regresión logística

\- Análisis discriminante lineal (LDA)

\- Análisis discriminante cuadrático (QDA)



El dataset contiene aproximadamente 145 000 observaciones provenientes de 49 estaciones meteorológicas australianas, con un marcado desbalance de clases: cerca del 77% de los días no presentan lluvia al día siguiente.



Este proyecto demuestra cómo la estadística multivariada permite capturar la estructura atmosférica subyacente del clima australiano y evaluar, de forma crítica, los límites de predicción en regiones áridas y de transición climática.



---



\## 2. Contexto climático de Australia y márgenes del clima



Australia constituye un caso de estudio excepcional para análisis climáticos debido a su diversidad de regímenes atmosféricos. En un solo continente coexisten climas tropicales, subtropicales, templados oceánicos y desérticos, lo que genera fuertes contrastes espaciales y temporales.



Desde el punto de vista de la clasificación de Köppen–Geiger (actualización 2021), los principales climas representados en el dataset son:



\- Cfa: subtropical húmedo, con lluvias distribuidas durante el año y temperaturas invernales superiores a 10 °C (ej. Sydney, Brisbane).

\- Aw: clima tropical de sabana, con estación seca marcada y precipitaciones anuales menores a 750 mm (ej. Darwin).

\- BSh/BWh: estepas y desiertos cálidos, caracterizados por una relación precipitación/evapotranspiración potencial menor a 0.5 (ej. Alice Springs).

\- Cfb: templado oceánico, con temperaturas moderadas y lluvias relativamente regulares (ej. Melbourne, Hobart).



El concepto central de este proyecto es el análisis de los \*\*márgenes climáticos\*\*, entendidos como regiones o condiciones donde el sistema atmosférico opera cerca de umbrales críticos. En estos márgenes, pequeñas variaciones en presión, humedad o temperatura pueden desencadenar cambios abruptos en la ocurrencia de lluvia.



A partir del análisis empírico del dataset WeatherAUS, se identifican los siguientes umbrales relevantes:



\- Rainfall:

&nbsp; - Percentil 5 ≈ 0 mm (condiciones secas)

&nbsp; - Percentil 95 ≈ 50 mm (eventos extremos)

\- MinTemp:

&nbsp; - Valores bajo 0 °C asociados a heladas y atmósfera estable

\- MaxTemp:

&nbsp; - Valores superiores a 40 °C asociados a olas de calor y baja probabilidad de lluvia

\- Pressure:

&nbsp; - Presiones menores a 1008 hPa asociadas a frentes lluviosos

\- Humidity3pm:

&nbsp; - Valores inferiores a 30% asociados a riesgo de incendios

&nbsp; - Valores superiores a 80% asociados a alta probabilidad de precipitación



Uno de los resultados más importantes del proyecto es que los modelos multivariados presentan mejor desempeño en climas húmedos o de transición (Cfa, Cfb) y un desempeño significativamente menor en climas áridos (BWh, BSh). Esto es consistente con la literatura climática y con el informe IPCC AR6, que destaca la dificultad de predecir precipitación en regiones dominadas por variabilidad episódica y forzantes de gran escala como ENSO.



---



\## 3. Dataset: WeatherAUS



Fuente principal: UCI Machine Learning Repository y Bureau of Meteorology (Australia).



Características generales:

\- Observaciones: aproximadamente 145 000

\- Estaciones meteorológicas: 49

\- Periodo temporal: 2013–2016

\- Variable objetivo: RainTomorrow (Yes / No)



Principales columnas del dataset:



| Variable | Tipo | % Missing | Descripción |

|--------|------|-----------|-------------|

| Date | Fecha | 0% | Fecha de observación |

| Location | Categórica | 0% | Estación meteorológica |

| MinTemp | Numérica | ~1% | Temperatura mínima |

| MaxTemp | Numérica | ~1% | Temperatura máxima |

| Rainfall | Numérica | ~2% | Precipitación diaria |

| Evaporation | Numérica | ~43% | Evaporación |

| Sunshine | Numérica | ~48% | Horas de sol |

| Humidity9am | Numérica | ~2% | Humedad relativa a las 9am |

| Humidity3pm | Numérica | ~3% | Humedad relativa a las 3pm |

| Pressure9am | Numérica | ~10% | Presión a las 9am |

| Pressure3pm | Numérica | ~10% | Presión a las 3pm |

| RainToday | Binaria | ~2% | Llovió hoy |

| RainTomorrow | Binaria | ~2% | Lloverá mañana |



Las variables Evaporation y Sunshine presentan un porcentaje elevado de valores faltantes, lo que condiciona su uso en modelos multivariados y obliga a estrategias de imputación cuidadosas.



---



\## 4. Análisis exploratorio de datos (EDA)



El EDA revela patrones consistentes con la física atmosférica:



\- Rainfall presenta una distribución altamente asimétrica, con una gran concentración de ceros.

\- Temperaturas muestran distribuciones aproximadamente unimodales y correlaciones muy altas entre sí.

\- Presión atmosférica presenta baja varianza pero altísima correlación entre mediciones de la mañana y la tarde.

\- Humedad relativa presenta correlación negativa con la temperatura máxima.

\- Existe un desbalance significativo en la variable objetivo: aproximadamente 77% de observaciones corresponden a “No Rain”.



Estos hallazgos justifican el uso de PCA para reducir redundancia y estabilizar los modelos de clasificación.



---



\## 5. Metodología



El pipeline metodológico del proyecto sigue los siguientes pasos:



1\. Carga y limpieza de datos con pandas.

2\. Análisis exploratorio de distribuciones, correlaciones y valores faltantes.

3\. Imputación de valores faltantes mediante media o KNN, según la variable.

4\. Codificación de variables categóricas (Location mediante one-hot, RainToday como binaria).

5\. Estandarización de variables numéricas.

6\. Aplicación de PCA reteniendo componentes que explican al menos el 85% de la varianza.

7\. Entrenamiento de modelos:

&nbsp;  - Regresión logística

&nbsp;  - LDA

&nbsp;  - QDA

8\. Evaluación mediante validación cruzada de 5 folds y métricas adecuadas para clases desbalanceadas.



El PCA identifica cuatro componentes principales con interpretación física clara:

\- PC1: componente térmico

\- PC2: estabilidad atmosférica (presión vs humedad)

\- PC3: viento frente a humedad

\- PC4: precipitación



---



\## 6. Resultados



Desempeño promedio en validación cruzada:



| Modelo | Accuracy | Recall (Yes) | Precision (Yes) | AUC |

|------|----------|--------------|-----------------|-----|

| Regresión logística + PCA | ~0.84 | ~0.72 | ~0.65 | ~0.85 |

| LDA | ~0.83 | ~0.70 | ~0.63 | ~0.83 |

| QDA | ~0.81 | ~0.68 | ~0.60 | ~0.82 |



La regresión logística combinada con PCA ofrece el mejor balance entre interpretabilidad y desempeño predictivo.



Las cargas del PCA muestran que Humidity3pm y Pressure9am son variables clave para anticipar la lluvia, lo cual concuerda con el conocimiento meteorológico.



---



\## 7. Análisis de márgenes climáticos



El análisis por locación revela que:



\- En estaciones costeras húmedas (Sydney, Brisbane), el modelo logra altos valores de recall.

\- En estaciones áridas del interior (Alice Springs), el modelo tiende a predecir sistemáticamente “No Rain”.

\- Los errores del modelo se concentran en días con Rainfall bajo pero no nulo (<1 mm), que representan transiciones difíciles de clasificar.



Este comportamiento confirma que los márgenes climáticos constituyen el principal límite para modelos estadísticos lineales.



---



\## 8. Instalación y reproducibilidad



Requisitos:

\- Python 3.9+

\- pandas

\- numpy

\- scikit-learn

\- statsmodels

\- matplotlib

\- seaborn

\- jupyter



Ejecución básica:



pip install -r requirements.txt

jupyter notebook ProyectoFinalEstadisticaMultivareada-1.ipynb



yaml

Copiar código



El notebook contiene todas las celdas necesarias para reproducir los resultados.



---



\## 9. Limitaciones



\- No se incorporan coordenadas geográficas ni relaciones espaciales explícitas.

\- La calidad de los datos depende de estaciones individuales.

\- Evaporation y Sunshine presentan missing data estructural.

\- No se incluyen índices climáticos de gran escala como ENSO.



