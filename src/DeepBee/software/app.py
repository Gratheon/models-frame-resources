# from flask import Flask
from detection_and_classification import run
# from platform import python_version

# app = Flask(__name__)

# print("Python version: ", python_version())

# @app.route("/")
# def hello():
#     # try:
#     #     print("running")
#     run()
#     #     print("Done")
#     # except Exception as e:
#     #     print(e)
#     return "Hello, World!"

if __name__ == "__main__":
    run()
    # app.run(host="0.0.0.0", port=8540, debug=True, threaded=False)


# import asyncio
# import aiomysql

# async def process_job(job):
#     # TODO: Process the job here
#     print(f"Processing job with id {job['id']}")

# async def read_jobs_from_db(pool):
#     async with pool.acquire() as conn:
#         async with conn.cursor() as cur:
#             await cur.execute("SELECT * FROM jobs WHERE status = 'pending'")
#             jobs = await cur.fetchall()
#             for job in jobs:
#                 await process_job(job)
#                 # Update the job status to "completed" in the database
#                 await cur.execute("UPDATE jobs SET status = 'completed' WHERE id = %s", job['id'])
#                 await conn.commit()

# async def main():
#     pool = await aiomysql.create_pool(
#         host='localhost', user='root', password='password', db='mydatabase')
#     while True:
#         await read_jobs_from_db(pool)
#         # Sleep for 1 second to avoid overloading the database with queries
#         await asyncio.sleep(1)

# if __name__ == '__main__':
#     asyncio.run(main())
