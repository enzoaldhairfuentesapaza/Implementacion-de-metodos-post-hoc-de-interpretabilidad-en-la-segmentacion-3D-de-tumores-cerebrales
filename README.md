# 🧠 Análisis comparativo de métodos post-hoc de interpretabilidad en la segmentación 3D de tumores cerebrales mediante MRI

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-EE4C2C.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-BraTS2020-green.svg)](https://www.med.upenn.edu/cbica/brats2020/)
[![UCSP](https://img.shields.io/badge/Universidad-UCSP-red.svg)](https://ucsp.edu.pe/)

Este repositorio contiene la implementación del proyecto académico **"Análisis comparativo de métodos post-hoc de interpretabilidad en la segmentación 3D de tumores cerebrales mediante MRI"**. El trabajo realiza una evaluación cuantitativa y cualitativa entre dos enfoques principales de Inteligencia Artificial Explicable (XAI) —**Grad-CAM** y **LIME 3D**— aplicados sobre una arquitectura **3D U-Net** entrenada con el dataset BraTS2020.

---

## 📌 Tabla de Contenidos
- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Metodología](#-metodología)
- [Resultados y Comparativa](#-resultados-y-comparativa)
- [Estructura del Repositorio](#-estructura-del-repositorio)
- [Instalación y Requisitos](#-instalación-y-requisitos)
- [Uso](#-uso)
- [Autores](#-autores)

---

## 🔍 Descripción del Proyecto

La opacidad de los modelos de Deep Learning (*cajas negras*) dificulta su adopción en la práctica médica[cite: 1]. En este trabajo analizamos y clasificamos métodos de interpretabilidad post-hoc para determinar qué técnica refleja de manera más fiel el comportamiento interno de una arquitectura de segmentación 3D en neuroimágenes multimodales[cite: 1].

Se evalúan dos filosofías opuestas de interpretabilidad[cite: 1]:
* **Grad-CAM 3D:** Basado en gradientes y activaciones de la última capa convolucional del decoder (`up4.conv`)[cite: 1].
* **LIME 3D:** Basado en perturbaciones sobre supervóxeles de $16^3$ vóxeles, evaluando cambios locales en la salida[cite: 1].

---

## 🛠️ Metodología

* **Dataset:** BraTS2020 (369 volúmenes MRI multimodales con secuencias T1, T1ce, T2 y FLAIR)[cite: 1].
* **Preprocesamiento:** Recorte y re-muestreo a $128^3$ vóxeles[cite: 1].
* **Arquitectura Objetivo:** 3D U-Net (entrenada con planificador de tasa de aprendizaje coseno, early stopping en época 22)[cite: 1].
* **Región de Evaluación XAI:** Clase *Enhancing Tumor* (ET)[cite: 1].
* **Métricas XAI:** Fidelidad (correlación de Spearman), Infidelidad, AUC Deletion, AUC Insertion y tiempo de ejecución[cite: 1].

---

## 📊 Resultados y Comparativa

### Rendimiento del Modelo Base (3D U-Net)
| Métrica | Valor[cite: 1] |
| :--- | :---: |
| Dice Promedio | 0.6299 |
| IoU Promedio | 0.5325 |
| Dice Whole Tumor (WT) | 0.6896 |
| Dice Tumor Core (TC) | 0.6128 |
| Dice Enhancing Tumor (ET) | 0.5873 |

### Comparativa Métodos XAI (Promedio sobre 20 muestras)
| Métrica | Grad-CAM | LIME 3D | Criterio de Calidad[cite: 1] |
| :--- | :---: | :---: | :--- |
| **Fidelidad** (Spearman) | **0.74** | 0.48 | Mayor concordancia con el comportamiento real (↑) |
| **Infidelidad** | **19.37** | 437.40 | Menor discrepancia ante perturbaciones (↓) |
| **AUC Deletion** | 0.00171 | **0.00044** | Caída rápida de confianza al eliminar zonas clave (↓) |
| **AUC Insertion** | 0.00421 | **0.00550** | Recuperación rápida al insertar zonas clave (↑) |
| **Tiempo medio (s)** | **12.42 s** | 464.54 s | Menor costo computacional (↓) |

* **Concordancia Espacial:** IoU(Grad-CAM $\cap$ LIME) = **0.489**[cite: 1].
* **Conclusión:** Grad-CAM genera mapas continuos, anatómicamente alineados y cuantitativamente más fieles con un tiempo de cómputo ~37 veces menor que LIME 3D[cite: 1].

---

## 📁 Estructura del Repositorio

```text
├── data/                  # Scripts para preparación y carga de datos BraTS2020
├── models/                # Definición del modelo 3D U-Net
├── xai/                   # Módulos de interpretabilidad (Grad-CAM 3D y LIME 3D)
│   ├── grad_cam.py
│   └── lime_3d.py
├── metrics/               # Implementación de Fidelidad, Infidelidad, Deletion y Insertion
├── utils/                 # Visualización y superposición de mapas de saliencia
├── evaluate_xai.py        # Script principal de evaluación comparativa
├── requirements.txt       # Lista de dependencias del entorno
└── README.md              # Documentación del proyecto

Autor
Enzo Aldhair Fuentes Apaza - Universidad Católica San Pablo (UCSP) -
