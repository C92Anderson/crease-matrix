import schedule
import time
import game_pipeline
import xG_update

def job():
    print("Updating...")

def run_games():
    print("Running games script")


#schedule.every().day.at("3:59").do(job)
#schedule.every().day.at("4:00").do(xG_update)
#schedule.every().day.at("5:00").do(game_pipeline)

while True:
    schedule.run_pending()
    time.sleep(1)