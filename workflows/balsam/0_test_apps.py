from balsam.api import ApplicationDefinition

class Hello(ApplicationDefinition):
    site = "polaris-testing"
    command_template = "echo Hello, {{ say_hello_to }}! CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES from Balsam"
Hello.sync()

class VecNorm(ApplicationDefinition):
    site = "polaris-testing"
    def run(self, vec):
        import json
        print(f"Hello {say_hello_to}")
        return sum(x**2 for x in json.loads(vec))**0.5
VecNorm.sync()
