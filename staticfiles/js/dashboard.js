(() => {
  const dataElement = document.getElementById("prediction-chart-data");
  const canvas = document.getElementById("predictionChart");
  if (!dataElement || !canvas || typeof Chart === "undefined") {
    return;
  }

  const counts = JSON.parse(dataElement.textContent);
  new Chart(canvas, {
    type: "doughnut",
    data: {
      labels: ["Positive model output", "Negative model output"],
      datasets: [{
        data: [counts.positive, counts.negative],
        backgroundColor: ["#1769aa", "#cbd8e3"],
        borderWidth: 0,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
        },
      },
    },
  });
})();
