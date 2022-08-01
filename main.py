from kivy.app import App
from kivy.uix.screenmanager import ScreenManager

from main_screen import MainScreen
from open_camera_screen import OpenCameraScreen
from test_screen import TestScreen
from train_screen import TrainScreen
from training_information_screen import TrainingInformationScreen


class MainApp(App):
    def build(self):
        screen_manager = ScreenManager()
        main_screen = MainScreen(name='main_screen')
        train_screen = TrainScreen(name='train_screen')
        test_screen = TestScreen(name='test_screen')
        open_camera_screen = OpenCameraScreen(name='open_camera_screen')
        training_information_screen = TrainingInformationScreen(name='training_information_screen')

        screen_manager.add_widget(main_screen)
        screen_manager.add_widget(train_screen)
        screen_manager.add_widget(test_screen)
        screen_manager.add_widget(open_camera_screen)
        screen_manager.add_widget(training_information_screen)

        return screen_manager


if __name__ == '__main__':
    MainApp().run()
