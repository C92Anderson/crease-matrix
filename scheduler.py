from crontab import CronTab

cron = CronTab(user=True)
job = cron.new(command='python /Users/colander1/PycharmProjects/GamePrediction/game_pipeline.py')

job.hour.on(10)

#cron.write()
job.clear()
cron.remove_all()

print(job.is_valid())

#12 10 * * * /usr/bin/python /Users/colander1/PycharmProjects/GamePrediction/xG_update.py
#0 5 * * * /usr/bin/python /Users/colander1/PycharmProjects/GamePrediction/game_pipeline.py

#https://stackabuse.com/scheduling-jobs-with-python-crontab/