# dl-fall22
Репозиторий для сдачи домашечек

## Как сдать домашнюю работу?

* Делаете форк репозитория
* Клонируете репозиторий к себе на компьютер
* Командой `git checkout -b hw-NN-github` создаете новую ветку в вашем **локальном** репозитории
    * Здесь `NN` - номер сдаваемой домашней работы, состоящий из двух разрядов
    * `github` - ван хэндл на гитхабе
    * Если номер домашней работы меньше 10, тогда в первый разряд вы ставите 0
    * Например, если я буду создавать ветку для второй домашней работы, я выполню команду `git checkout -b hw-02-wdywbac`
* Вносите необходимые изменения в папке соответствующего занятия (например, добавляете jupyter-ноутбук с выполненным заданием)
* Проверяете, какие файлы вы изменили при помощи команды `git status`
* Сохраняете изменения в вашей ветке:
    * `git add [пути_к_добавляемым_файлам]`
    * Добавить все файлы можно командой `git add .`
* Создаете коммит: `git commit -m "текст коммита"`. Коммиты фиксируют сохранённые изменения в рамках конкретной ветки. Коммиты с изменениями не синхронизируются между разными ветками автоматически.
* Отправляете изменения из вашего локального репозитория в хранилище github
    * Если это новая ветка, которую вы ещё не "пушили" в github, то вам нужна команда `git push --set-upstream origin название_вашей_ветки`
    * Если вы уже ранее отправляли изменения в ветке, то можно выполнить команду `git push`
* Создаете PR (pull-request) из вашей ветки в основной репозиторий
    * Название PR'а должно совпадать с названием ветки

Если вам нужно сдать другую домашнюю работу, вам необходимо 
* Переключиться на основную ветку командой `git checkout main`
* Подтянуть новые изменения из исходного репозитория, если они есть (https://stackoverflow.com/questions/3903817/pull-new-updates-from-original-github-repository-into-forked-github-repository)
* Создать новую ветку по вышеописанному процессу
* Для сдачи лабораторной работы используйте номер 00

## FAQ
### У меня не получается сдать домашнюю работу, что мне делать?
Напишите об этом в чат или преподавателю
### Я сдал домашнюю работу вовремя, но назвал PR неправильно
Название PR'а можно редактировать. Если вы исправите название даже после дедлайна, то штрафа не последует, если изначально задание было сдано вовремя
