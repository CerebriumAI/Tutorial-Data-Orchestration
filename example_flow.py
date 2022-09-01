from prefect import flow, task


## Tasks
@task
def foo():
    print("foo 1")

@task
def foo_name(name):
    print(f"foo {name}")


## Flow
@flow
def foo_flow(names):
    foo()
    for name in names:
        foo_name.submit(name)


# Call the flow
foo_flow(["2 - electric boogaloo", "3 - gigantic spree"])
