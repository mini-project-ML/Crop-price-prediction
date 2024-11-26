from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
import joblib
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "super_secret_key"

# Create a database for user registration
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Load the prediction model
best_model = joblib.load('best_model_pipeline.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Load dataset for min/max price lookup
data = pd.read_csv("agricrop.csv")
data['commodity_name'] = data['commodity_name'].str.lower()
data['state'] = data['state'].str.lower()
data['district'] = data['district'].str.lower()
data['market'] = data['market'].str.lower()

# Home route: User Registration
@app.route('/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        hashed_password = generate_password_hash(password)

        conn = sqlite3.connect("users.db")
        c = conn.cursor()

        try:
            c.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", (name, email, hashed_password))
            conn.commit()
            session['user'] = name  # Save user in session
            return redirect(url_for('login'))  # Redirect to login page after registration
        except sqlite3.IntegrityError:
            return render_template("register.html", error_message="Email already registered.")
        finally:
            conn.close()

    return render_template("register.html")

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if 'email' in request.form and 'password' in request.form:
            email = request.form['email']
            password = request.form['password']

            conn = sqlite3.connect("users.db")
            c = conn.cursor()

            c.execute("SELECT * FROM users WHERE email = ?", (email,))
            user = c.fetchone()

            if user and check_password_hash(user[3], password):  # Assuming password is at index 3
                session['user'] = user[1]  # Save username to session
                return redirect(url_for('prediction'))
            else:
                return render_template("login.html", error_message="Invalid credentials.")
        else:
            return "Error: Missing email or password in the form."
    
    return render_template("login.html")


# Prediction route
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if 'user' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in

    commodity_name = state = district = market = ""  # Default values for the inputs
    predicted_price = None

    if request.method == 'POST':
        commodity_name = request.form['commodity_name'].strip().lower()
        state = request.form['state'].strip().lower()
        district = request.form['district'].strip().lower()
        market = request.form['market'].strip().lower()

        try:
            encoded_commodity = label_encoders['commodity_name'].transform([commodity_name])[0]
            encoded_state = label_encoders['state'].transform([state])[0]
            encoded_district = label_encoders['district'].transform([district])[0]
            encoded_market = label_encoders['market'].transform([market])[0]

            matching_rows = data[
                (data['commodity_name'] == commodity_name) &
                (data['state'] == state) &
                (data['district'] == district) &
                (data['market'] == market)
            ]

            if not matching_rows.empty:
                avg_min_price = matching_rows['min_price'].mean()
                avg_max_price = matching_rows['max_price'].mean()
            else:
                avg_min_price = data['min_price'].mean()
                avg_max_price = data['max_price'].mean()

            input_data = pd.DataFrame([[encoded_commodity, encoded_state, encoded_district, encoded_market, avg_min_price, avg_max_price]],
                                      columns=['commodity_name', 'state', 'district', 'market', 'min_price', 'max_price'])
            predicted_price = best_model.predict(input_data)[0]

        except KeyError as e:
            return render_template('prediction.html', error="Invalid input. Please check your entries.", 
                                   commodity_name=commodity_name, state=state, district=district, market=market)

    return render_template('prediction.html', 
                           predicted_price=f"{predicted_price:.2f}" if predicted_price else None, 
                           commodity_name=commodity_name, 
                           state=state, 
                           district=district, 
                           market=market)
# Logout
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('register'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
