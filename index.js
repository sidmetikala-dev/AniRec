async function fetchData(){

    const username = document.getElementById("username").value.trim();
    const out = document.getElementById("out");
    out.textContent = "Loading...";

    try{

        const response = await fetch("/api/recs", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username }),
        });

        if(!response.ok){
            throw new Error("Could not fetch resource");
        }

        const data = await response.json();

        // titles = Object.values(data);
        out.textContent = data.join("\n");
        console.log("From Python:", data);

    }
    catch(error) {
        out.textContent = `Error: ${error.message}`;
        console.error(error);
    }
}