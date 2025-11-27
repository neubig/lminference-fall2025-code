# Request script

1. Each system will be tested with the same simulation, generated with something like the function in `generate_simulation.py`. The final simulation is not guaranteed to have the same overall request frequency and batch shape distribution, but it will be within the same ballpark and at least not significantly harder. jsons provided for convenience
2. `python make_requests.py` to run a simulation. Pre-loaded with a lightweight 1 minute sample -- in reality the simulation will be longer and most likely denser