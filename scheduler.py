from crontab import CronTab

cron = CronTab(user=True)
job = cron.new(command='python game_pipeline.py')
job.hour.every(1)

cron.write()

#12 10 * * * /usr/bin/python /Users/colander1/PycharmProjects/GamePrediction/xG_update.py
#0 5 * * * /usr/bin/python /Users/colander1/PycharmProjects/GamePrediction/game_pipeline.py

#https://stackabuse.com/scheduling-jobs-with-python-crontab/