import cv2
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen, SlideTransition

import face_recognition


class TrainScreen(Screen):
    def __init__(self, **kwargs):
        super(TrainScreen, self).__init__(**kwargs)

        self.label = Label(text='Do you want to take images or load it', size_hint=(1, .8))
        self.open_camera = Button(text='Open Camera', size_hint=(1, .1), on_release=self.open_camera_pressed)
        self.load_images = Button(text='Load images', size_hint=(1, .1))

        self.back = Button(text='Back', size_hint=(1, .1), on_release=self.back_callback)

        button_layout = BoxLayout(orientation='horizontal')
        button_layout.add_widget(self.open_camera)
        button_layout.add_widget(self.load_images)

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.back)
        layout.add_widget(self.label)
        layout.add_widget(button_layout)

        self.add_widget(layout)

    def open_camera_pressed(self, instance):
        self.manager.transition = SlideTransition(direction='left')
        self.manager.current = 'open_camera_screen'

        i = face_recognition.find_screen(self.manager.children, 'open_camera_screen')

        self.manager.children[i].capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.manager.children[i].update, 1.0 / 33.0)

    def back_callback(self, instance):
        self.manager.transition = SlideTransition(direction='right')
        self.manager.current = 'main_screen'
