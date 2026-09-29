#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>
#include <iomanip>

using namespace std;

// ─── Student struct ───────────────────────────────────────
struct Student {
    string name;
    double dsa;
    double projects;
    double iq;
    double attendance;
    int label;  // 1 = placed, 0 = not placed
};

// ─── Normalization stats (computed from training data) ────
struct NormStats {
    double dsa_min,        dsa_max;
    double projects_min,   projects_max;
    double iq_min,         iq_max;
    double attendance_min, attendance_max;
};

// ─── Model weights ────────────────────────────────────────
struct Model {
    double w_dsa        = 0.0;
    double w_projects   = 0.0;
    double w_iq         = 0.0;
    double w_attendance = 0.0;
    double bias         = 0.0;
};

// ─── Read CSV file ────────────────────────────────────────
vector<Student> loadCSV(const string& filename) {
    vector<Student> data;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error: could not open " << filename << "\n";
        return data;
    }

    string line;
    getline(file, line); // skip header row

    while (getline(file, line)) {
        stringstream ss(line);
        string token;
        Student s;

        getline(ss, s.name, ',');
        getline(ss, token, ',');
         s.dsa        = stod(token);
        getline(ss, token, ','); 
        s.projects   = stod(token);
        getline(ss, token, ','); 
        s.iq         = stod(token);
        getline(ss, token, ','); 
        s.attendance = stod(token);
        getline(ss, token, ','); 
        s.label      = stoi(token);

        data.push_back(s);
    }

    cout << "Loaded " << data.size() << " students from " << filename << "\n";
    return data;
}

// ─── Compute min and max from training data ───────────────
// We do NOT hardcode these values
// We calculate from actual data so normalization is fair
NormStats computeNormStats(const vector<Student>& data) {
    NormStats stats;

    // initialize with first student's values
    stats.dsa_min        = stats.dsa_max        = data[0].dsa;
    stats.projects_min   = stats.projects_max   = data[0].projects;
    stats.iq_min         = stats.iq_max         = data[0].iq;
    stats.attendance_min = stats.attendance_max = data[0].attendance;

    for (const Student& s : data) {
        stats.dsa_min        = min(stats.dsa_min,        s.dsa);
        stats.dsa_max        = max(stats.dsa_max,        s.dsa);
        stats.projects_min   = min(stats.projects_min,   s.projects);
        stats.projects_max   = max(stats.projects_max,   s.projects);
        stats.iq_min         = min(stats.iq_min,         s.iq);
        stats.iq_max         = max(stats.iq_max,         s.iq);
        stats.attendance_min = min(stats.attendance_min, s.attendance);
        stats.attendance_max = max(stats.attendance_max, s.attendance);
    }

    cout << "\n─── Normalization Stats (from training data) ───\n";
    cout << "DSA:        [" << stats.dsa_min        << ", " << stats.dsa_max        << "]\n";
    cout << "Projects:   [" << stats.projects_min   << ", " << stats.projects_max   << "]\n";
    cout << "IQ:         [" << stats.iq_min         << ", " << stats.iq_max         << "]\n";
    cout << "Attendance: [" << stats.attendance_min << ", " << stats.attendance_max << "]\n";

    return stats;
}

// ─── Normalize a single value ─────────────────────────────
// formula: (value - min) / (max - min)
// result is always between 0 and 1
double normalize(double value, double min_val, double max_val) {
    if (max_val == min_val) return 0.0; // avoid divide by zero
    return (value - min_val) / (max_val - min_val);
}

// ─── Sigmoid ──────────────────────────────────────────────
double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

// ─── Forward pass ─────────────────────────────────────────
double predict(const Model& m, const Student& s, const NormStats& stats) {
    // normalize each feature using stats from training data
    double n_dsa        = normalize(s.dsa,        stats.dsa_min,        stats.dsa_max);
    double n_projects   = normalize(s.projects,   stats.projects_min,   stats.projects_max);
    double n_iq         = normalize(s.iq,         stats.iq_min,         stats.iq_max);
    double n_attendance = normalize(s.attendance, stats.attendance_min, stats.attendance_max);

    double z = m.w_dsa        * n_dsa
             + m.w_projects   * n_projects
             + m.w_iq         * n_iq
             + m.w_attendance * n_attendance
             + m.bias;

    return sigmoid(z);
}

// ─── Binary cross entropy loss ────────────────────────────
double computeLoss(double P, int y) {
    double eps = 1e-9;
    return -(y * log(P + eps) + (1 - y) * log(1 - P + eps));
}

// ─── Train ────────────────────────────────────────────────
void train(Model& m, const vector<Student>& data,
           const NormStats& stats, double lr, int epochs) {

    cout << "\n─── Training ───\n";

    for (int epoch = 0; epoch < epochs; epoch++) {

        double totalLoss = 0.0;

        for (const Student& s : data) {

            // forward pass
            double P   = predict(m, s, stats);
            totalLoss += computeLoss(P, s.label);

            // error
            double error = P - s.label;

            // update weights immediately after each student
            m.w_dsa        -= lr * error * normalize(s.dsa,        stats.dsa_min,        stats.dsa_max);
            m.w_projects   -= lr * error * normalize(s.projects,   stats.projects_min,   stats.projects_max);
            m.w_iq         -= lr * error * normalize(s.iq,         stats.iq_min,         stats.iq_max);
            m.w_attendance -= lr * error * normalize(s.attendance, stats.attendance_min, stats.attendance_max);
            m.bias         -= lr * error;
        }

        if (epoch % 100 == 0) {
            cout << "Epoch " << setw(4) << epoch
                 << "  |  Loss: " << fixed << setprecision(4)
                 << totalLoss / data.size() << "\n";
        }
    }
}

// ─── Accuracy on training data ────────────────────────────
void evaluate(const Model& m, const vector<Student>& data,
              const NormStats& stats) {
    int correct = 0;
    for (const Student& s : data) {
        double P      = predict(m, s, stats);
        int predicted = (P >= 0.5) ? 1 : 0;
        if (predicted == s.label) correct++;
    }
    double accuracy = (double)correct / data.size() * 100.0;
    cout << "\n─── Accuracy on training data ───\n";
    cout << "Correct: " << correct << " / " << data.size()
         << "  (" << fixed << setprecision(2) << accuracy << "%)\n";
}

// ─── Main ─────────────────────────────────────────────────
int main() {

    // load data from CSV
    vector<Student> data = loadCSV("students.csv");
    if (data.empty()) return 1;

    // compute normalization stats FROM training data
    // no hardcoding — model figures out min/max itself
    NormStats stats = computeNormStats(data);

    // initialize model
    Model model;

    // train
    train(model, data, stats, 0.1, 1000);

    // print final weights
    cout << "\n─── Final Weights ───\n";
    cout << "w_dsa:        " << model.w_dsa        << "\n";
    cout << "w_projects:   " << model.w_projects   << "\n";
    cout << "w_iq:         " << model.w_iq         << "\n";
    cout << "w_attendance: " << model.w_attendance << "\n";
    cout << "bias:         " << model.bias         << "\n";

    // evaluate accuracy
    evaluate(model, data, stats);
    return 0;
}