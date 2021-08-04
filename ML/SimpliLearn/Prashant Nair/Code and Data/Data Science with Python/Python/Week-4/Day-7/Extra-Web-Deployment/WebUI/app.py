from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import pickle
import numpy as np
# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

model = pickle.load(open('HRSalaryPredictor.model','rb'))

class ReusableForm(Form):
    name = TextField('Name:', validators=[validators.required()])
    
    @app.route("/", methods=['GET', 'POST'])
    def hello():
        form = ReusableForm(request.form)
        
        print(form.errors)
        if request.method == 'POST':
            data = request.form['yexperience']
            if data == '':
                flash('Please enter years of experience')
            else:
                experience=float(data)
                if experience < 0:
                    flash('Not a valid year of experience')
                else:
                    exp = np.array([[experience]])
                    salary = model.predict(exp)
                    sal = "%.2f" % salary
                    print('The salary for {} years of experience is {}'.format(experience,sal))
                    flash('The salary for {} years of experience is {}'.format(experience,sal))
        
        if form.validate():
        # Save the comment here.
            flash('The salary for {} years of experience is {}'.format(experience,sal))
        else:
            pass
        
        return render_template('hello.html', form=form)

if __name__ == "__main__":
    app.run()