from utils.custom_flask import CustomFlask
from flask import redirect
from sparcassist.pages import sparcassist_pages

debug = False

app = CustomFlask(__name__)
app.jinja_env.filters['zip'] = zip
# app.register_blueprint(faxplain_page)
app.register_blueprint(sparcassist_pages)


@app.route('/')
def main():
    return redirect('/sparcassist', code=200)