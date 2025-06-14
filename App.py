from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import os
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'supersecretkey'

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def run_forecast(file_path, output_path):
    df = pd.read_csv(file_path, encoding='latin-1')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d-%m-%Y %H:%M', errors='coerce')

    daily_sales = df.groupby(df['InvoiceDate'].dt.date)['Quantity'].sum().reset_index()
    daily_sales.columns = ['ds', 'y']

    model = Prophet()
    model.fit(daily_sales)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Save forecast plot
    fig = model.plot(forecast)
    plot_path = os.path.join(output_path, 'forecast_plot.png')
    fig.savefig(plot_path)
    plt.close(fig)

    # Save forecast CSV
    csv_path = os.path.join(output_path, 'sales_forecast.csv')
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(csv_path, index=False)

    return plot_path, csv_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part in the request')
        return redirect(url_for('main'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('main'))

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        plot_path, csv_path = run_forecast(filepath, OUTPUT_FOLDER)
        flash(f'Forecast generated! You can download the results below.')
        return render_template('main.html', plot_url='/' + plot_path, csv_url='/' + csv_path)

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
