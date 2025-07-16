document.addEventListener("DOMContentLoaded", function () {
    let countdownInterval; // Global countdown interval variable

    // Function to update the status
    function updateStatus() {
        fetch("/get_status")
            .then(response => response.json())
            .then(data => {
                const statusEl = document.getElementById("status");
                const countdownEl = document.getElementById("countdown");
                const timerEl = document.getElementById("timer");
                const detectedNameEl = document.getElementById("detected-name");

                if (data.intruder) {
                    // Update status with intruder details
                    statusEl.innerText = `🔴 Detected: ${data.detected || 'Unknown'} at ${data.timestamp || 'N/A'}`;
                    statusEl.style.color = "red"; // Highlight intruder detection

                    // Show countdown timer
                    countdownEl.style.display = "block";
                    let timeLeft = data.countdown !== null ? data.countdown : 15;
                    timerEl.innerText = timeLeft;

                    // Show detected person's name
                    detectedNameEl.innerText = `👤 ${data.detected || 'Unknown'}`;
                    detectedNameEl.style.display = "block";

                    // Prevent multiple countdown intervals
                    clearInterval(countdownInterval);
                    countdownInterval = setInterval(() => {
                        timeLeft--;
                        timerEl.innerText = timeLeft;

                        if (timeLeft <= 0) {
                            clearInterval(countdownInterval);
                            countdownEl.style.display = "none";
                            detectedNameEl.style.display = "none"; // Hide detected name
                        }
                    }, 1000);
                } else {
                    // Update status to "No Intruder Detected"
                    statusEl.innerText = "✅ No Intruder Detected";
                    statusEl.style.color = "green"; // Indicate all clear

                    // Hide countdown and detected name
                    countdownEl.style.display = "none";
                    detectedNameEl.style.display = "none";

                    // Stop countdown if no intruder detected
                    clearInterval(countdownInterval);
                }
            })
            .catch(error => {
                console.error("⚠️ Error fetching status:", error);
                const statusEl = document.getElementById("status");
                statusEl.innerText = "⚠️ Error fetching status. Please try again.";
                statusEl.style.color = "orange"; // Indicate a warning
            });
    }

    // Function to refresh the video feed
    function refreshFeed() {
        const videoFeedEl = document.getElementById("videoFeed");
        videoFeedEl.src = `${videoFeedEl.src.split("?")[0]}?t=${new Date().getTime()}`; // Add a timestamp to force refresh
        alert("🔄 Video feed refreshed!");
    }

    // Function to trigger an alert manually
    function forceAlert() {
        fetch("/trigger_alert", { method: "POST" })
            .then(response => {
                if (response.ok) {
                    alert("⚠️ Alert triggered successfully!");
                } else {
                    alert("⚠️ Failed to trigger alert. Please try again.");
                }
            })
            .catch(error => {
                console.error("⚠️ Error triggering alert:", error);
                alert("⚠️ Error triggering alert. Please check the console for details.");
            });
    }

    // Update status every 2 seconds
    setInterval(updateStatus, 2000);

    // Attach event listeners to buttons
    document.querySelector(".btn").addEventListener("click", refreshFeed);
    document.querySelector(".alert-btn").addEventListener("click", forceAlert);
});