from flask import Flask, request, jsonify
import modules

app = Flask(__name__)

https_match = r'^https:\/\/[^\s"]+$'

@app.route('/process', methods=['POST'])
def process_data():
    try:
        data = request.json
        query = data.get('input', '')
        sites_required = data.get('sites_required', 20)  # Default to 20 if not provided
        gemini_response = modules.main(query, https_match, sites_required)
        response = {
            "result": gemini_response,
        }
        return jsonify(response)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    try:
        modules.clear_cache()  # Call the clear_cache function from the modules
        return jsonify({"message": "Cache cleared successfully."}), 200
    except Exception as e:
        print(f"Error clearing cache: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
