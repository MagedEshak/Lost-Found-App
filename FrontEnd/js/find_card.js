$(document).ready(function () {
  $("#register-card-2-form").on("submit", async function (e) {
    e.preventDefault();

    $(".error-msg").text("");

    const CardID = $("#national_id").val().trim();
    const Government = $("#government").val().trim();
    const Center = $("#center").val().trim();
    const Street = $("#street").val().trim();
    const FileImage = $("#fileInput")[0].files[0];
    const FinderEmail = $("#email").val().trim();

    let hasError = false;

    if (!CardID) {
      $("#err_national_id").text("National ID is required");
      hasError = true;
    } else if (!/^\d{14}$/.test(CardID)) {
      $("#err_national_id").text("National ID must be 14 digits");
      hasError = true;
    }

    if (!Government) {
      $("#err_government").text("Government is required");
      hasError = true;
    }

    if (!Center) {
      $("#err_center").text("City is required");
      hasError = true;
    }

    if (!Street) {
      $("#err_street").text("Street is required");
      hasError = true;
    }

    if (!FileImage) {
      $("#err_fileInput").text("Please upload a card image");
      hasError = true;
    }

    if (!FinderEmail || !/\S+@\S+\.\S+/.test(FinderEmail)) {
      $("#err_email").text("Valid email is required");
      hasError = true;
    }

    if (hasError) return;

    try {
      const token = localStorage.getItem("token");
      const formData = new FormData();
      formData.append("CardID", CardID);
      formData.append("Government", Government);
      formData.append("Center", Center);
      formData.append("Street", Street);
      formData.append("CardPhoto", FileImage);
      formData.append("FinderEmail", FinderEmail);
      console.log(FileImage);

      const response = await fetch(
        "http://localhost:5194/api/Find_Card/Add_Find_Card",
        {
          method: "POST",
          body: formData,
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      if (response.ok) {
        $("#register-card-2-form")[0].reset();
        ShowBootstrapToast(
          "Card added successfully. What would you like to do next?",
          "success",
          true
        );
      } else {
        const errorData = await response.text();
        ShowBootstrapToast("Error: " + errorData, "danger");
      }
    } catch (error) {
      console.error("Error:", error);
      ShowBootstrapToast("Unexpected error occurred", "danger");
    }
  });
});

// -----------------------  TostBox (Popup Message)    ----------------------------------
window.ShowBootstrapToast = function (
  message,
  type = "Info",
  withButtons = false
) {
  const toastId = "custom-toast-" + Date.now();
  const toastHTML = `
        <div id="${toastId}" class="toast align-items-center text-white bg-${type.toLowerCase()} border-0" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="d-flex">
                <div class="toast-body w-100">
                    ${message}
                    ${
                      withButtons
                        ? `
                    <div class="mt-2 pt-2 border-top d-flex justify-content-end gap-2">
                        <button type="button" class="btn btn-light btn-sm" id="btn-add-new">Add Another</button>
                        <button type="button" class="btn btn-outline-light btn-sm" id="btn-go-home">Go Home</button>
                    </div>`
                        : ""
                    }
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        </div>
    `;

  let toastContainer = document.getElementById("toast-container");
  if (!toastContainer) {
    toastContainer = document.createElement("div");
    toastContainer.id = "toast-container";
    toastContainer.className =
      "toast-container position-fixed bottom-0 end-0 p-3";
    document.body.appendChild(toastContainer);
  }

  toastContainer.innerHTML += toastHTML;

  const toastElement = document.getElementById(toastId);
  const toast = new bootstrap.Toast(toastElement, { delay: 7000 });
  toast.show();

  if (withButtons) {
    toastElement
      .querySelector("#btn-add-new")
      .addEventListener("click", function () {
        $("#register-card-2-form")[0].reset();
        $("#preview").attr(
          "src",
          "../../images/id-card-illustration_23-2147829294.avif"
        );
        $(".error-msg").text("");
        toast.hide();
      });

    toastElement
      .querySelector("#btn-go-home")
      .addEventListener("click", function () {
        window.location.replace("../../index.html");
      });
  }

  toastElement.addEventListener("hidden.bs.toast", function () {
    toastElement.remove();
  });
};
