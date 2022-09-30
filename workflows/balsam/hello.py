from balsam.api import ApplicationDefinition

class Hello(ApplicationDefinition):
    site = "polaris_tutorial"
    command_template = "echo Hello, {{ say_hello_to }}! CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
Hello.sync()
