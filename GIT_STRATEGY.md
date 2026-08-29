# Estrategia de Git

Documento de la estrategia de control de versiones utilizada en el proyecto
(Punto 6 del entregable Curso II - MLE).

## Resumen

Se sigue un flujo inspirado en **GitHub Flow**: una rama base estable (`main`),
una rama de integracion (`development`) y trabajo incremental mediante commits
con mensajes descriptivos. Toda integracion a `main` se realiza a traves de
**Pull Requests** revisados y mergeados.

## Ramas

| Rama | Proposito | Regla |
|---|---|---|
| `main` | Version estable y entregable | Solo recibe cambios via Pull Request mergeada |
| `development` | Integracion y trabajo en curso | Rama de trabajo sobre la que se prepara el release |

Flujo basico:

1. Se crea una rama corta desde `development` (o se trabaja directamente sobre
   `development` para cambios pequenos).
2. Se hacen commits con mensajes claros y atomicos.
3. Se envia un Pull Request de `development -> main`.
4. Se revisa el PR y se fusiona con merge (squash/merge segun el caso).
5. Se etiqueta el commit final con la version (`v1.0.0`) y se crea el Release.

## Convenciones de commits

Mensajes en formato `<tipo>: <descripcion>`:

| Tipo | Uso |
|---|---|
| `feat` | Nueva funcionalidad |
| `fix` | Correccion de errores |
| `docs` | Documentacion (README, notebooks, docstrings) |
| `chore` | Tareas de mantenimiento (limpieza, configuracion) |
| `refactor` | Reestructuracion sin cambio de comportamiento |
| `test` | Pruebas |

Ejemplos del historial:

```
feat: DagsHub MLflow tracking with model artifacts and Model Registry
docs: rewrite README following the deliverable spec
docs: git strategy documentation
chore: remove leftover churn-analysis template files
```

## Versionado y releases

- Se usa **versionado semantico** (`vMAYOR.MENOR.PATCH`).
- La entrega final del curso esta en la version **`v1.0.0`**.
- El flujo de release es:
  1. Congelar `main` en el commit de entrega.
  2. Crear tag anotado `v1.0.0` apuntando a ese commit.
  3. Crear un **GitHub Release** `v1.0.0` con notas que describen el contenido de la
     version (proceso, modelos, evidencia).
- Video de referencia oficial:
  https://docs.github.com/es/repositories/releasing-projects-on-github/managing-releases-in-a-repository

Pull Request evidencia: `#1` y `#2` (development -> main), mergeadas.

## Remotes

| Remote | URL | Uso |
|---|---|---|
| `origin` | `https://github.com/0DaikiHajime0/Proyecto2.git` | Repositorio oficial de entrega (GitHub) |
| `dagshub` | `https://dagshub.com/0DaikiHajime0/Proyecto2.git` | Espejo del codigo + evidencia MLflow (experimentos y Model Registry) |

`origin` es el repositorio de entregable; `dagshub` duplica el codigo y aloja el
tracking de MLflow (experimentos con metricas, parametros y artefactos) en
`https://dagshub.com/0DaikiHajime0/Proyecto2.mlflow`.

## Buenas practicas aplicadas

- `.gitignore` para excluir datos derivados, credenciales y archivos temporales.
- Commits atomicos y descriptivos (mensajes con prefijo de tipo).
- Pull Requests para integrar a `main` (al menos una cerrada exitosamente).
- Tags + Release para la entrega en version `1.0.0`.
- Uso de `.gitkeep` para carpetas provisionales de datos (`data/*`, `models/`).
- Mirrors remotos para evidencia complementaria (DagsHub).