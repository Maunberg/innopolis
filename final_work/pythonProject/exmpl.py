class CustomTemplateWindow(BoxLayout):
    def __init__(self, **kwargs):
        super(CustomTemplateWindow, self).__init__(**kwargs)
        self.clear_widgets()
        self.orientation = 'vertical'

        label1 = Label(text='Введите название шаблона:', size_hint=(1, 0.1))
        self.add_widget(label1)

        input_field1 = TextInput(multiline=True, size_hint=(1, 0.1))
        self.add_widget(input_field1)

        label2 = Label(text='Введите название таблицы:', size_hint=(1, 0.1))
        self.add_widget(label2)

        input_field2 = TextInput(multiline=True, size_hint=(1, 0.1))
        self.add_widget(input_field2)

        button_fill = Button(text='Заполнить', size_hint=(1, 0.1))
        button_fill.bind(on_press=lambda instance: FuncDoc.my_doc(input_field1.text, input_field2.text))
        self.add_widget(button_fill)

        button4 = Button(text='Выход', size_hint=(1, 0.1))
        button4.bind(on_press=lambda instance: self.cancel())
        self.add_widget(button4)

    def cancel(self, instance):
        self.root_window.remove_widget(self.root_window.children[0])


class MyApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')

        label = Label(text='Пожалуйста, выберите способ заполнения документов и пропишите ниже путь к файлу:',
                      size_hint=(1, 0.3))
        layout.add_widget(label)

        input_field = TextInput(multiline=False, size_hint=(1, 0.1))
        layout.add_widget(input_field)

        button1 = Button(text='Заполнение документов для абитуриентов', size_hint=(1, 0.1))
        button1.bind(on_press=lambda instance: self.process_input(input_field.text, '1'))
        layout.add_widget(button1)

        button2 = Button(text='Заполнение документов для должников', size_hint=(1, 0.1))
        button2.bind(on_press=lambda instance: self.process_input(input_field.text, '2'))
        layout.add_widget(button2)

        button3 = Button(text='Пользовательский шаблон', size_hint=(1, 0.1))
        button3.bind(on_press=lambda instance: self.show_custom_template_window(input_field.text))
        layout.add_widget(button3)

        button4 = Button(text='Выход', size_hint=(1, 0.1))
        button4.bind(on_press=lambda instance: self.cancel())
        layout.add_widget(button4)

        return layout

    def process_input(self, file_path, option):
        if option == '1':
            FuncDoc.abitur_doc(file_path)
        elif option == '2':
            FuncDoc.debt_doc(file_path)

    def show_custom_template_window(self, file_path):
        self.root_window.remove_widget(self.root_window.children[0])
        custom_window = CustomTemplateWindow()
        self.root_window.add_widget(custom_window)

    def cancel(self, instance):
        self.root_window.remove_widget(self.root_window.children[0])


if __name__ == '__main__':
    MyApp().run()