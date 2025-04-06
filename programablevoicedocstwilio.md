Programmable Voice Quickstart for Python


With just a few lines of code, your Python application can make and receive phone calls with Twilio Programmable Voice.

This Python quickstart will teach you how to do this using our REST API, the Twilio Python helper library, and Python's Flask microframework
 to ease development.

In this quickstart, you will learn how to:

Sign up for Twilio and get your first voice-enabled Twilio phone number
Set up your development environment to make and receive phone calls
Make an outbound phone call which plays an MP3
Receive and respond to an inbound phone call which reads a message to the caller using Text to Speech
Prefer to get started by watching a video? Check out our video on how to place and receive phone calls with Python on YouTube
.

Sign up for Twilio and get a phone number


(information)
Info
If you already have a Twilio account and a voice-enabled Twilio phone number you're all set here! Log in
 then feel free to jump to the next step.

Before you can make a phone call from Python, you'll need a Twilio account. Sign up here
 to get your free trial account or log in
 to an account you already have.

The next thing you'll need is a voice-capable Twilio phone number
. If you don't currently own a Twilio phone number with voice call functionality, you'll need to purchase one. After navigating to the Buy a Number
 page, check the "Voice" box and click "Search."

Search for a voice-enabled phone number.
Expand image
You'll then see a list of available phone numbers and their capabilities. Find a number that suits your fancy and click "Buy" to add it to your account.

Purchase a voice-enabled phone number from Twilio.
Expand image
Now that you have a Twilio account and a programmable phone number, you have the basic tools you need to make a phone call.

Next, we'll install Twilio's official Python helper library to help communicate with Twilio's APIs.

Install Python and the Twilio helper library


(information)
Info
If you've gone through one of our Python quickstarts already and have Python and the Twilio Python helper library installed, you can skip this step and get straight to making your first phone call.

To make your first phone call with Twilio, you'll need to have Python and the Twilio Python helper library installed.

Install Python


If you're using a Mac or Linux machine, you probably already have Python installed. You can check this by opening up a terminal and running the following command:


Copy code block
python --version
You should see something like:


Copy code block
$ python --version
Python 3.9  # Python 3.7+ is okay, too
Windows users can follow this excellent tutorial for installing Python on Windows
.

Twilio's Python SDK only supports Python 3.7+.

Install the Twilio Python helper library


The easiest way to install the library is using pip
. Just run this in the terminal:


Copy code block
pip install twilio
If you get the error pip: command not found, you can use easy_install to install the Twilio helper library by running this in your terminal:


Copy code block
easy_install twilio
For a manual installation, you can download the source code (ZIP)
 for twilio-python and then install the library by running:


Copy code block
python setup.py install
in the folder containing the twilio-python library.

And with that, it's time to write some code.

Make an outgoing phone call with Python


Now that we have Python and twilio-python installed, we can make an outgoing phone call with a single API request from the Twilio phone number we just purchased. Create a new file called make_call.py and type or paste in this code sample.

Make a phone call using Twilio

Python

Report code block

Copy code block
# Download the helper library from https://www.twilio.com/docs/python/install
import os
from twilio.rest import Client

# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
account_sid = os.environ["TWILIO_ACCOUNT_SID"]
auth_token = os.environ["TWILIO_AUTH_TOKEN"]
client = Client(account_sid, auth_token)

call = client.calls.create(
    url="http://demo.twilio.com/docs/voice.xml",
    to="+15558675310",
    from_="+15017122661",
)

print(call.sid)
Output

Copy output
{
  "account_sid": "ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
  "annotation": null,
  "answered_by": null,
  "api_version": "2010-04-01",
  "caller_name": null,
  "date_created": "Tue, 31 Aug 2010 20:36:28 +0000",
  "date_updated": "Tue, 31 Aug 2010 20:36:44 +0000",
  "direction": "inbound",
  "duration": "15",
  "end_time": "Tue, 31 Aug 2010 20:36:44 +0000",
  "forwarded_from": "+141586753093",
  "from": "+15017122661",
  "from_formatted": "(415) 867-5308",
  "group_sid": null,
  "parent_call_sid": null,
  "phone_number_sid": "PNaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "price": "-0.03000",
  "price_unit": "USD",
  "sid": "CAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "start_time": "Tue, 31 Aug 2010 20:36:29 +0000",
  "status": "completed",
  "subresource_uris": {
    "notifications": "/2010-04-01/Accounts/ACaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Calls/CAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Notifications.json",
    "recordings": "/2010-04-01/Accounts/ACaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Calls/CAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Recordings.json",
    "payments": "/2010-04-01/Accounts/ACaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Calls/CAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Payments.json",
    "events": "/2010-04-01/Accounts/ACaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Calls/CAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Events.json",
    "siprec": "/2010-04-01/Accounts/ACaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Calls/CAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Siprec.json",
    "streams": "/2010-04-01/Accounts/ACaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Calls/CAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Streams.json",
    "transcriptions": "/2010-04-01/Accounts/ACaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Calls/CAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Transcriptions.json",
    "user_defined_message_subscriptions": "/2010-04-01/Accounts/ACaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Calls/CAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/UserDefinedMessageSubscriptions.json",
    "user_defined_messages": "/2010-04-01/Accounts/ACaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Calls/CAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/UserDefinedMessages.json"
  },
  "to": "+15558675310",
  "to_formatted": "(415) 867-5309",
  "trunk_sid": null,
  "uri": "/2010-04-01/Accounts/ACaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Calls/CAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.json",
  "queue_time": "1000"
}
This code starts a phone call between the two phone numbers that we pass as arguments. The 'from' number is our Twilio number, and the 'to' number is who we want to call.

The URL argument points to some TwiML, which tells Twilio what to do next when our recipient answers their phone. This TwiML tells Twilio to read a message using text to speech and then play an MP3.

Before this code will work, though, we need to edit it a little to work with your Twilio account.

Replace the placeholder credential values


Swap the placeholder values for account_sid and auth_token with your personal Twilio credentials.

Go to https://www.twilio.com/console
 and log in. On this page, you'll find your unique Account SID and Auth Token, which you'll need any time you send messages through the Twilio client like this. You can reveal your auth token by clicking on the eyeball icon:

Reveal Your Auth Token.
Expand image
Open make_call.py and replace the values for account_sid and auth_token with your unique values.

(error)
Danger
Please note: it's okay to hardcode your credentials when getting started, but you should use environment variables to keep them secret before deploying to production. Check out how to set environment variables
 for more information.

Replace the to and from_ phone numbers


Remember that voice-enabled phone number you bought just a few minutes ago? Go ahead and replace the existing from_ number with that one, making sure to use E.164 formatting:

[+][country code][phone number including area code]

Next, replace the to phone number with your mobile phone number. This can be any phone number that can receive calls, but it's a good idea to test with your phone so that you can see the magic happen! As above, you should use E.164 formatting for this value.

Save your changes and run the script from your terminal:


Copy code block
python make_call.py
That's it! Your phone should ring with a call from your Twilio number, and you'll hear our short message for you. 😉

(warning)
Warning
If you're using a Twilio trial account, outgoing phone calls are limited to phone numbers you have verified with Twilio. Phone numbers can be verified via your Twilio Console's Verified Caller IDs
. For other trial account restrictions and limitations, check out our guide on how to work with your free Twilio trial account.

Next, we'll learn how to respond to a call made to your Twilio phone number. First, we'll need to get a Flask server up and running.

Install Flask and set up your development environment


To handle incoming phone calls we'll need a lightweight web application to accept incoming HTTP requests from Twilio. We'll use Flask
 for this quickstart, but you can use your choice of web framework to make and receive phone calls from your applications.

For instructions on setting up Flask on Windows, check out this guide
.

Install pip and virtualenv


To install Flask and set up our development environment, we'll need two tools: pip
 to install Flask and virtualenv
 to create a unique sandbox for this project. If you already have these tools installed, you can skip to the next section.

Pip comes pre-packaged with Python 3.4+, so if you're on a recent version of Python, you don't need to install anything new. If you're on an earlier version, never fear: pip is included in virtualenv. So let's install virtualenv!

If you're using Python 2.4, run the following command in your terminal:


Copy code block
easy_install virtualenv
If you're using Python 2.5-2.7, run the following command in your terminal, specifying your version number:


Copy code block
easy_install-2.7 virtualenv
Replace the 2.7 with 2.5 or 2.6 if you have that version installed.

To install virtualenv with Python 3.4+:


Copy code block
# If you get 'permission denied' errors try running "sudo python" instead of "python"
pip install virtualenv
If you get any errors in this step, check out these tips for debugging.

Create and activate your virtual environment


Once you have virtualenv installed, use your terminal to navigate to the directory you're using for this quickstart and create a virtual environment:


Copy code block
cd Documents/my_quickstart_folder
virtualenv --no-site-packages .
Now, activate the virtual environment:


Copy code block
source bin/activate
You can verify that your virtualenv is running by looking at your terminal: you should see the name of your enclosing directory. It will look something like this:


Copy code block
(my_quickstart_folder)USER:~ user$
To learn more about virtualenv or create a custom environment path, see this thorough guide
.

Install dependencies


Now we're ready to install Flask. Create a file called requirements.txt and add the following lines to it:


Copy code block
Flask>=0.12
twilio~=6.0.0
Then install both of these packages with pip in your terminal:


Copy code block
pip install -r requirements.txt
Test everything from scratch


First, make sure your virtualenv is activated:


Copy code block
cd Documents/my_quickstart_folder
source bin/activate     # On Windows, use .\bin\activate.bat
Then, create and open a file called answer_phone.py and add these lines:

'Hello, World' Flask application

Python

Report code block

Copy code block
from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

if __name__ == "__main__":
    app.run()
Output

Copy output
$ python app.py
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
Now it's time to try running it. In your terminal, type:


Copy code block
python answer_phone.py
You should see:


Copy code block
$ python answer_phone.py
* Running on http://127.0.0.1:5000/
Navigate to http://localhost:5000
 in your browser. You should see a "Hello World!" message. You're ready to create your first Twilio Programmable Voice app.

(warning)
Warning
If you encountered any issues or want instructions on setting up your environment with an older Python version (<3.4), check out our full guide to setting up a local Python dev environment.

Allow Twilio to talk to your Flask application


We're about to enhance our small Flask application to accept incoming phone calls. But before we do that, we need to make sure Twilio can talk to our local development environment.

Most Twilio services use webhooks to communicate with your application. When Twilio receives a phone call, for example, it reaches out to a URL in your application for instructions on how to handle the call.

When you're working on your Flask application in your development environment, your app is only reachable by other programs on your computer and Twilio won't be able to talk to it. We need to solve this problem by making your application accessible over the internet.

While there are a lot of ways to do this, like deploying your application to Heroku or AWS, you'll probably want a less laborious way to test your Twilio application. For a lightweight way to make your app available on the internet, we recommend a tool called ngrok. Once started, ngrok provides a unique URL on the ngrok.io domain which forwards incoming requests to your local development environment.

How ngrok helps Twilio reach your local server.
Expand image
If you don't already use ngrok, head over to their download page
 and grab the appropriate binary for your operating system. Once downloaded, unzip the package.

If you're working on a Mac or Linux, you're all set. If you're on Windows, follow our guide on how to install and configure ngrok on Windows. For more info on ngrok, including some great tips and tricks, check out this in-depth blog post
.

Once downloaded, start that Hello World application we made previously:


Copy code block
python answer_phone.py
Your local application must be running locally for ngrok to do its magic.

Then open a new terminal tab or window and start ngrok with this command:


Copy code block
./ngrok http 5000
If your local server is running on a different port, replace 5000 with the correct port number.

You should see output similar to this:

Ngrok server terminal output.
Expand image
Copy your public URL from this output and paste it into your browser. You should see your Flask application's "Hello World!" message.

Respond to incoming calls with Twilio


When your Twilio number receives an incoming phone call, it sends an HTTP request to your server asking for instructions on what to do next. Once you receive the request, you can tell Twilio how to respond to the call.

For this quickstart, we'll have our Flask app reply to answer the phone call and say a short message to the caller. Open up answer_phone.py again and update the code to look like this code sample:

Respond to an incoming call with a brief message
Respond to an incoming request from Twilio with instructions on how to handle the call


Copy code block
from flask import Flask
from twilio.twiml.voice_response import VoiceResponse

app = Flask(__name__)


@app.route("/answer", methods=['GET', 'POST'])
def answer_call():
    """Respond to incoming phone calls with a brief message."""
    # Start our TwiML response
    resp = VoiceResponse()

    # Read a message aloud to the caller
    resp.say("Thank you for calling! Have a great day.", voice='Polly.Amy')

    return str(resp)

if __name__ == "__main__":
    app.run(debug=True)
Save the file and restart your app with


Copy code block
python answer_phone.py
You should now be able to open a web browser to http://localhost:5000/answer
. If you view the page source code, you should see the following text:


Copy code block
<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Amy">Thank you for calling! Have a great day.</Say>
</Response>
This source code is TwiML XML generated by your code with the help of the Twilio helper library.

Double-check that ngrok is still running on localhost with the same port as before. Now Twilio will be able to find your application. There's just one last thing we need before we're ready to call your app: we need to tell Twilio where to send its request.

Configure your webhook URL


For Twilio to know where to look, you need to configure your Twilio phone number to call your webhook URL whenever a new message comes in.

Log in to Twilio.com and go to the Console's Numbers page
.
Click on your voice-enabled phone number.
Find the "Voice & Fax" section. Make sure the "Accept Incoming" selection is set to "Voice Calls." The default "Configure With" selection is what you'll need: "Webhooks/TwiML...".
In the "A Call Comes In" section, select "Webhook" and paste in the URL you want to use, appending your '/answer' route:
Configure your Voice webhook with your ngrok URL.
Expand image
Save your changes - you're ready!

Test your application


As long as your localhost and the ngrok servers are up and running, we're ready for the fun part - testing our new Flask application!

Make a phone call from your mobile phone to your Twilio phone number. You should see an HTTP request in your ngrok console. Your Flask app will process the incoming request and respond with your TwiML. Then you'll hear your message once the call connects.

