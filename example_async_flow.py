from prefect import flow, task
import asyncio

## Tasks
@task
def foo():
    print("foo 1")


@task
async def foo_name(name):
    print(f"foo 2 - {name}")


## Flow
@flow
async def foo_flow(names):
    await asyncio.gather(*[foo_name(name) for name in names])
    foo()
    await foo_name.submit(names)


# Call the flow
if __name__ == "__main__":
    asyncio.run(foo_flow(["2 - electric boogaloo", "3 - gigantic spree"]))
