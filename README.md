## Main description

Данный репозиторий содержит код ЭВМ для задач:
- (`lexical`) Программа для выявления лексических ошибок в тексте сочинений по английскому языку
- (`logic`) Программа для выявления нарушений логики или логических ошибок в последовательности рассуждений
- (`history`) Программа для выявления содержательных ошибок и упоминаний исторических событий в тексте сочинений по истории


Каждая задача содержит набор сервисов, которые могут понадобиться для решения, обращения к сервисам происходит с помощью HTTP-API запросы:
- `lexical`
    - `basic_reader` - аннотатор
    - `gector` - сервис для решения задачи
- `logic`
    - `basic_reader` - аннотатор
    - `sense_blocks_detector` - сервис для решения задачи
- `history`
    - `basic_reader` - аннотатор
    - `morphosyntactic_parser` - аннотатор
    - `ner` - аннотатор
    - `event_detector` - сервис для решения задачи
    - `history_role_skill` - сервис для решения задачи
    - `date_check_skill` - сервис для решения задачи
    - `history_k3_skill` - сервис для решения задачи

## Requirements

* `docker`
* `docker-compose`

## QuickStart

Перед запуском любой задачи в начале необходимо развернуть окружение.
Для раворачивания окружения используется `docker-compose`. В следующем примере используется переменная среды `$TASK`.
`$TASK` - может примать следующие значения:
- `lexical`
- `logic`
- `history`

В начале необходимо создать `gpus.yml`, который будет содержать описание для тех сервисов, которым необходим GPU.

``` bash
tools/init_gpus_file.sh $TASK.yml
```
Для того, чтобы назначить правильные id GPU, необходимо после создания файла `gpus.yml` отредактировать переменную `CUDA_VISIBLE_DEVICES=""` . После этого можно развернуть окружение для задачи
``` bash
docker-compose -f $TASK.yml -f gpus.yml up -d --build
```
### Extended guide

Для того чтобы получить информацию о запущенных сервисах:

``` bash
docker-compose -f $TASK.yml -f gpus.yml ps
```
Результат работы команды:
``` 
              Name                           Command                State         Ports  
-----------------------------------------------------------------------------------------
basic_reader                       /bin/sh -c ./server_run.sh    Up (healthy)            
```

Для того, чтобы запустить тест для любого сервиса необходимо выполнить команду:

``` bash
docker-compose exec $service python test_server.py
```
Во время теста файлы `services/*/*/test_data/*_input.json` отравляются через http-запрос в сервис, а ответы сравниваются с `services/*/*/test_data/*_output.json`. Эти же файлы можно использовать для примера как те данные, которые используется сервисами для работы

Получение лога работы сервиса:

``` bash
docker-compose logs -f $service
```

### Solver/Annotator

Шаблон выходных/входных данные сервиса: 
``` json
{
    "input_data": [
        {
            "raw_input": "RAW_TEXT_1",
            "annotations": {
                "basic_reader": "OUTPUT_DATA",
                "ANNOTATOR_1_NAME": "OUTPUT_DATA",
                "ANNOTATOR_2_NAME": "OUTPUT_DATA"
            },
            "solvers": [
                {"SOLVER_1_NAME": "OUTPUT_DATA"},
                {"SOLVER_2_NAME": "OUTPUT_DATA"}
    ]
        },
    ]

```

### basic_reader 
Пример выходных данных

``` json
{
    "annotations": {
        "sections": [
            {
                "link": [
                    "#LINK_LABEL"
                ],
                "link_information": [
                    "#TEXT_WITH_TAG"
                ],
                "raw_text": "TEXT",
                "raw_type": "SECTION_TYPE",
                "start_span": "INDEX",
                "end_span": "INDEX",
                "text": "TEXT",
                "type": "SECTION_TYPE"
            }
        ],
        "mistakes": [
            {
                "link": [
                    "#LINK_LABEL"
                ],
                "link_information": [
                    "#TEXT_WITH_TAG"
                ],
                "raw_text": "TEXT",
                "raw_type": "ERROR_TYPE",
                "start_span": "INDEX",
                "end_span": "INDEX",
                "text": "TEXT",
                "type": "ERROR_TYPE"
            }
        ],        
    },
    "clear_essay": "TEXT",
    "clear_essay_sentences": [
       [
          "SENTENCE"
       ]
    ],
    "criteria": {
        "К_NAME": "K_VAL"
    },
    "raw_essay": "TEXT",
    "subject": "SUBJECT_NAME",
    "meta": {
        "тема": "TOPIC",
        "год": "YEAR",
        "тест": "TEST_TYPE",
        "эксперт": "EXPERT_ID",
        "класс": "GRADE",
        "линия": "LINE",
        "отрывок": "TEXT"
    }
}

```

