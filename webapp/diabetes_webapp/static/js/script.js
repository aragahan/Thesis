function required()
    {
      var input1 = document.forms["myForm"]["age"].value;
      var input2 = document.forms["myForm"]["glucose_level"].value;
      var input3 = document.forms["myForm"]["diastolic_bp"].value;
      var input4 = document.forms["myForm"]["systolic_bp"].value;
      var input5 = document.forms["myForm"]["heart_rate"].value;
      var input6 = document.forms["myForm"]["body_temp"].value;
      var input7 = document.forms["myForm"]["SPO2"].value;
      var input8 = document.forms["myForm"]["sweating"].value;
      var input9 = document.forms["myForm"]["shivering"].value;
      var fileinput = document.getElementById("csv_file");
      var filePath = fileinput.value
      var allowedExtensions = /(\.csv)$/i;
      
      // Error checker: There should be no blank inputs, either manually input values or upload a file
      // Checks for blank inputs and no file uploaded
      if ((input1 == "" || input2 == "" || input3 == "" || input4 == "" || input5 == ""
      || input6 == "" || input7 == "" || input8 == "" || input9 == "") && fileinput.files.length == 0)
      {
        alert("Either input ALL fields or upload a CSV file to proceed, please try again.");
        return false;
      }
      

      // Check if there are BOTH inputs and file upload
      // Checks if all fields have inputs and if a file is uploaded
      else if (fileinput.files.length !== 0 )
      {
          if (input1 != "" || input2 != "" || input3 != "" || input4 != "" || input5 != "" || 
            input6 != "" || input7 != "" || input8 != "" || input9 != ""){
            alert("There CANNOT be both inputted fields AND an uploaded file, only choose one method.");
            return false;
          }
          
          //Checks for invalid input type
          else if (!allowedExtensions.exec(filePath))
          {
            alert("Invalid file type, please only upload CSV files.");
            return false;
          }
      }
      
      // It is now assumed that we will enter values on the fields and NOT used the file input
      // All input boxes have inputs, now we check if the inputs are correct
      // Makes sure that there is no uploaded file
      else if (fileinput.files.length == 0)
      { 
        // Check if the input queries were right
        console.log(input1);
        console.log(input2);
        console.log(input3);
        console.log(input4);
        console.log(input5);
        console.log(input6);
        console.log(input7);
        console.log(input8);
        console.log(input9);
        
        // Change the inputs to integer
        input1 = parseInt(input1);
        input2 = parseInt(input2);
        input3 = parseInt(input3);
        input4 = parseInt(input4);
        input5 = parseInt(input5);
        input7 = parseInt(input7);
        input8 = parseInt(input8);
        input9 = parseInt(input9);

        
        // Checks if age is an integer and within a valid range
        if (input1 < 1 || input1 > 100) {
          alert("Input for AGE must be a valid integer from 1 to 100.");
          return false;
        }
        
        // Checks if blood glucose level is an integer and within a valid range
        else if (Number.isInteger(input2) == false || input2 < 50 || input2 > 260){
          alert("Input for BLOOD GLUCOSE LEVEL must be a valid integer from 50 to 260.");
          return false;
        }

        // Checks if diastolic bp is an integer and within a valid range
        else if (Number.isInteger(input3) == false || input3 < 60 || input3 > 120){
          alert("Input for DIASTOLIC BLOOD PRESSURE must be a valid integer from 60 to 120.");
          return false;
        }

        // Checks if systolic bp is an integer and within a valid range
        else if (Number.isInteger(input4) == false || input4 < 100 || input4 > 150){
          alert("Input for SYSTOLIC BLOOD PRESSURE must be a valid integer from 100 to 150.");
          return false;
        }

        // Checks if diastolic > systolic
        else if (input3 > input4 || input3 == input4){
          alert("Input for DIASTOLIC BLOOD PRESSURE must be LESS than SYSTOLIC BLOOD PRESSURE");
          return false;
        }

        // Checks if heart rate is an integer and within a valid range
        else if (Number.isInteger(input5) == false || input5 < 60 || input5 > 120){
          alert("Input for HEART RATE must be a valid integer from 60 to 120.");
          return false;
        }

        // Checks if body temp is within a valid range
        else if (input6 < 35 || input6 > 40){
          alert("Input for BODY TEMPERATURE must be from 35 to 40.");
          return false;
        }

        // Checks if SPO2 is an integer and within a valid range
        else if (Number.isInteger(input7) == false || input7 < 90 || input7 > 100){
          alert("Input for SPO2 must be from 90 to 100.");
          return false;
        }

        // Checks if input for sweating is valid
        else if (Number.isInteger(input8) == false || input8 > 1 || input8 < 0) {
          alert("Input for SWEATING must be either integer 0 or 1.");
          return false;
        }

        // Check if input for shivering is valid
        else if (Number.isInteger(input9) == false || input9 > 1 || input9 < 0){
          alert("Input for SHIVERING must be either integer 0 or 1.");
          return false;
        }
        
      }
      
      return true;
    }