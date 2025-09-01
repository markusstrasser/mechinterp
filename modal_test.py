import modal
from adder import adder

app = modal.App("example-test")
img = (modal.Image.debian_slim()
       .add_local_python_source("adder")
       )


@app.function(image=img)
def square(x):
    print("This code is running on a remote worker!", adder(x, 1))
    return x**2


@app.local_entrypoint()
def main():
    print("the square is", square.remote(42))

