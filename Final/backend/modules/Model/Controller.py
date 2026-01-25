from modules.Model import Services


async def get(type): return await Services.fetch_data(type)