#include <bits/stdc++.h>

using namespace std;

int main() {

    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int numDevices, numConnections;
    cin >> numDevices >> numConnections;

    vector<vector<pair<int, int>>> edgeWeights(numDevices);
    int i, j, w;
    for(int k = 0; k < numConnections; ++k) {
        cin >> i >> j >> w;
        edgeWeights[i].push_back({j, w});
        edgeWeights[j].push_back({i, w});
    }

    vector<bool> visited(numDevices, false);
    vector<int> distances(numDevices, INT_MAX);

    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;

    distances[0] = 0;
    pq.push({0, 0});


    while (!pq.empty()) {
        auto [bestDistance, currentNode] = pq.top();
        pq.pop();

        if (visited[currentNode]) continue;

        visited[currentNode] = true;

        for (const auto& [neighbor, weight] : edgeWeights[currentNode]) {
            if (!visited[neighbor] && weight < distances[neighbor]) {
                distances[neighbor] = weight;
                pq.push({distances[neighbor], neighbor});
            }
        }
    }

    int totalWeight = 0;
    for (int d : distances) {
        totalWeight += d;
    }

    cout << totalWeight << endl;

    return 0;
}
