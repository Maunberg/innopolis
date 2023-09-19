import DEF as d
import pandas as pd
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
import pickle

class Body(BoxLayout):
    def __init__(self, **kwargs):
        super(Body, self).__init__(**kwargs)
        self.clear_widgets()
        self.orientation = 'vertical'

        global name
        label = Label(text=f'Вывод вашего файла: {name}',
                      size_hint=(1, 0.1))
        self.add_widget(label)

        global data
        label1 = Label(text=data.head().to_string(), size_hint=(1, 0.2))
        self.add_widget(label1)


        button_made = Button(text='Приступить к обработке данных', size_hint=(1, 0.1))
        button_made.bind(on_press=lambda instance: self.data_made())
        self.add_widget(button_made)

        button_fit = Button(text='Приступить к подбору модели', size_hint=(1, 0.1))
        button_fit.bind(on_press=lambda instance: self.model_made())
        self.add_widget(button_fit)

    def data_made(self):
        self.clear_widgets()
        self.orientation = 'vertical'

        label1 = Label(text='Обработка пропусков', size_hint=(1, 0.1))
        self.add_widget(label1)

        nulla = Button(text='Автоматическая', size_hint=(1, 0.1))
        nulla.bind(on_press=lambda instance: self.work_withdata('1'))
        self.add_widget(nulla)

        nullu = Button(text='Ручная', size_hint=(1, 0.1))
        nullu.bind(on_press=self.show_popup)
        self.add_widget(nullu)

        label2 = Label(text='Обработка дубликатов', size_hint=(1, 0.1))
        self.add_widget(label2)

        dupa = Button(text='Автоматическая', size_hint=(1, 0.1))
        dupa.bind(on_press=lambda instance: self.work_withdata('2'))
        self.add_widget(dupa)

        dupu = Button(text='Ручная', size_hint=(1, 0.1))
        dupu.bind(on_press=self.show_popup)
        # dupu.bind(on_press=lambda instance: self.work_withdata('3'))
        self.add_widget(dupu)

        typa = Button(text='Далее', size_hint=(1, 0.1))
        typa.bind(on_press=lambda instance: self.data_made2())
        self.add_widget(typa)

    def show_popup(self, instance):
        # Создание всплывающего окна
        popup = Popup(title='Всплывающее окно',
                      size_hint=(None, None), size=(400, 200))

        # Контент всплывающего окна
        popup_content = BoxLayout(orientation='vertical')
        popup_content.add_widget(Label(text='Функция в разработке'))

        # Кнопка "Закрыть"
        button_close = Button(text='Закрыть')
        button_close.bind(on_press=popup.dismiss)
        popup_content.add_widget(button_close)

        popup.content = popup_content

        # Отображение всплывающего окна
        popup.open()

    def work_withdata(self, menu):
        global data
        if menu == '1':
            data = d.autofillna(data)

        elif menu == '2':
            data = d.duplcheck(data)

        elif menu == '3':
            pass

        elif menu == '4':
            data = self.sub

    def show_popup2(self, instance):
        # Создание всплывающего окна
        popup = Popup(title='Список столбцов')

        # Контент всплывающего окна
        popup_content = BoxLayout(orientation='vertical')

        global data
        self.list = '\n'.join(data.columns)
        popup_content.add_widget(Label(text=self.list))

        # Кнопка "Закрыть"
        button_close = Button(text='Закрыть')
        button_close.bind(on_press=popup.dismiss)
        popup_content.add_widget(button_close)

        popup.content = popup_content

        # Отображение всплывающего окна
        popup.open()

    def data_made2(self):
        self.clear_widgets()
        self.orientation = 'vertical'

        global data
        data, self.errors = d.dtypes(data)
        # data = d.findcorr(data)

        if self.errors == True:
            label3 = Label(
                text='Во время обработки данных возникли ошибки. Возможно, в таблице есть "грязные данные". Работа с ними будет в следующих версиях')
            self.add_widget(label3)

        self.sub, self.flag, self.num = d.analitic(data)
        if self.flag == True:
            label2 = Label(text=f'Слишком много аномалий: {self.num * 100 // 1}%', size_hint=(1, 0.1))
            self.add_widget(label2)

            typa = Button(text='Удалить их', size_hint=(1, 0.1))
            typa.bind(on_press=lambda instance: self.work_withdata('3'))
            self.add_widget(typa)
        else:
            data = self.sub.copy()

            self.model_made()

    def model_made(self):

        self.clear_widgets()
        self.orientation = 'vertical'

        label1 = Label(text='Выберите столбец:', size_hint=(1, 0.1))
        self.add_widget(label1)

        self.input_field = TextInput(multiline=False, size_hint=(1, 0.1))
        self.add_widget(self.input_field)

        columnslist = Button(text='Показать список столбцов', size_hint=(1, 0.1))
        columnslist.bind(on_press=self.show_popup2)
        self.add_widget(columnslist)

        columnslist = Button(text='Приступить к подбору модели', size_hint=(1, 0.1))
        columnslist.bind(on_press=lambda instance: self.model())
        self.add_widget(columnslist)

    def model(self):
        target = self.input_field.text
        self.clear_widgets()
        self.orientation = 'vertical'

        global X_train, X_test, y_train, y_test
        X_train, X_test, y_train, y_test = d.dtsplit(data, target)

        que = y_train.dtype in ['int64', 'float64']
        que2 = len(y_train.unique())>10
        if que and que2:
            label1 = Label(text='Ваша задача: Регрессия', size_hint=(1, 0.1))
            self.add_widget(label1)

            mod1 = Button(text='Запустить автоматический подбор модели:', size_hint=(1, 0.1))
            mod1.bind(on_press=lambda instance: self.work_withdata('1'))
            self.add_widget(mod1)

            mod2 = Button(text='Запустить автоматический перебор моделей:', size_hint=(1, 0.1))
            mod2.bind(on_press=lambda instance: self.work_withdata('2'))
            self.add_widget(mod2)

        else:
            label1 = Label(text='Ваша задача: Классификация', size_hint=(1, 0.1))
            self.add_widget(label1)

            mod1 = Button(text='Запустить автоматический подбор модели:', size_hint=(1, 0.1))
            mod1.bind(on_press=lambda instance: self.work_with_model('3'))
            self.add_widget(mod1)

            mod2 = Button(text='Запустить автоматический перебор моделей:', size_hint=(1, 0.1))
            mod2.bind(on_press=lambda instance: self.work_with_model('4'))
            self.add_widget(mod2)

        label2 = Label(text='Подбор и обучение модели может длиться до 4-5 часов.', size_hint=(1, 0.2))
        self.add_widget(label2)

        label3 = Label(text='Вы можете задать место сохранения модели, чтобы не потерять её:', size_hint=(1, 0.2))
        self.add_widget(label3)

        self.label4 = TextInput(multiline=False, size_hint=(1, 0.1))
        self.add_widget(self.label4)


    def save_model(self, wayto):
        wayto = wayto.replace('\\', '/').replace('"', '')
        with open(wayto, 'wb') as file:
            pickle.dump(best_model, file)

    def work_with_model(self, menu):
        self.wayto = (self.label4.text).replace('\\', '/').replace('"', '')
        global best_model, results_data
        if menu == '3':
            self.clear_widgets()
            self.orientation = 'vertical'
            label1 = Label(text='Результаты:...', size_hint=(1, 0.2))
            self.add_widget(label1)
            best_model, results_data = d.simple_search(X_train, X_test, y_train, y_test)
            if len(self.wayto)>0:
                with open(self.wayto, 'wb') as file:
                    pickle.dump(best_model, file)
            label2 = Label(text=str(results_data), size_hint=(1, 0.2))
            self.add_widget(label2)
        elif menu == '4':
            self.clear_widgets()
            self.orientation = 'vertical'
            label1 = Label(text='Результаты:...', size_hint=(1, 0.2))
            self.add_widget(label1)
            best_model, results_data = d.model_search(X_train, X_test, y_train, y_test)
            if len(self.wayto)>0:
                with open(self.wayto, 'wb') as file:
                    pickle.dump(best_model, file)
            label2 = Label(text=results_data.to_string(), size_hint=(1, 0.2))
            self.add_widget(label2)

        label3 = Label(text='Вы можете задать место сохранения модели, чтобы не потерять её:', size_hint=(1, 0.2))
        self.add_widget(label3)

        self.label4 = TextInput(multiline=False, size_hint=(1, 0.1))
        self.add_widget(self.label4)

        mod1 = Button(text='Сохранить', size_hint=(1, 0.1))
        mod1.bind(on_press=lambda instance: self.save_model(self.label4.text))
        self.add_widget(mod1)





class Start(App):

    def build(self):
        layout = BoxLayout(orientation='vertical')
        label = Label(text='Пожалуйста, пропишите путь к файлу:',
                      size_hint=(1, 0.3))
        layout.add_widget(label)

        input_field = TextInput(multiline=False, size_hint=(1, 0.1))
        layout.add_widget(input_field)

        button1 = Button(text='Загрузить файл', size_hint=(1, 0.1))
        button1.bind(on_press=lambda instance: self.body())
        layout.add_widget(button1)
        return layout

    def body(self):
        self.way = (self.root_window.children[0].children[1].text).replace('\\', '/').replace('"', '')
        global data
        global name
        name = (self.way).split('/')[-1]
        data = d.load(self.way)

        self.root_window.remove_widget(self.root_window.children[0])
        custom_window = Body()
        self.root_window.add_widget(custom_window)

    def cancel(self, instance):
        self.root_window.remove_widget(self.root_window.children[0])


if __name__ == '__main__':
    Start().run()
