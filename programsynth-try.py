import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    # absolute_minimal.py
    import itertools

    # Your "programs" are just function compositions
    primitives = {
        'up': lambda p: (p[0], p[1]+1),
        'down': lambda p: (p[0], p[1]-1),
        'left': lambda p: (p[0]-1, p[1]),
        'right': lambda p: (p[0]+1, p[1])
    }

    def search_for_square():
        """Find program that draws a square"""
        start = (0, 0)
        target_path = [(0,0), (1,0), (1,1), (0,1), (0,0)]

        # Try all combinations up to length 8
        for length in range(1, 9):
            for combo in itertools.product(primitives.keys(), repeat=length):
                pos = start
                path = [pos]
                for move in combo:
                    pos = primitives[move](pos)
                    path.append(pos)

                if path[:5] == target_path:
                    print(f"Found square program: {combo}")
                    # Now "compress" this into a new primitive!
                    primitives['square'] = lambda p: combo
                    return combo

    # This IS library learning at its core!
    search_for_square()
    return (primitives,)


@app.cell
def _(primitives):
    primitives
    return


@app.cell
def _():


    return


if __name__ == "__main__":
    app.run()
