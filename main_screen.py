import cv2
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen, SlideTransition

import face_recognition


class MainScreen(Screen):
    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        self.label = Label(text='FACE RECOGNITION APP', font_size=32)

        train_button = Button(text='Train', font_size=32, on_release=self.train_button_pressed)
        test_button = Button(text='Test', font_size=32, on_release=self.test_button_pressed)

        button_layout = BoxLayout(orientation='horizontal')
        button_layout.add_widget(train_button)
        button_layout.add_widget(test_button)

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.label)
        layout.add_widget(button_layout)
        self.add_widget(layout)

    def train_button_pressed(self, *args):
        self.manager.transition = SlideTransition(direction='left')
        self.manager.current = 'train_screen'

    def test_button_pressed(self, *args):
        self.manager.transition = SlideTransition(direction='left')
        self.manager.current = 'test_screen'
        i = face_recognition.find_screen(self.manager.children, 'test_screen')

        self.manager.children[i].model = face_recognition.load_the_model('siamesemodel')
        self.manager.children[i].capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.manager.children[i].update, 1.0 / 33.0)
