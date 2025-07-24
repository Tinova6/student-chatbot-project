from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger
import datetime

scheduler = AsyncIOScheduler()

async def schedule_test_notification(websocket, test_subject: str, user_id: str):
    """
    This function would theoretically send a WebSocket notification when a test is due.
    In a real application, you'd integrate this with your WebSocket manager.
    For now, it just prints.
    """
    notification_message = f"🔔 Reminder: Your {test_subject} test is now ready!"
    print(f"[{datetime.datetime.now()}] Sending notification to user {user_id}: {notification_message}")
    # You would send this via the active WebSocket connection for the user
    await websocket.send_json({"type": "notification", "message": notification_message})

def add_scheduled_job(func, run_date: datetime.datetime, *args, **kwargs):
    """
    Adds a job to the scheduler.
    func: The asynchronous function to run (e.g., schedule_test_notification)
    run_date: The datetime object when the job should run.
    *args, **kwargs: Arguments to pass to the function.
    """
    scheduler.add_job(func, DateTrigger(run_date=run_date), args=args, kwargs=kwargs)

if not scheduler.running:
    scheduler.start()