for dir in models/ycb/*/google_16k; do
    # Remove the trailing slash to get the folder name
    folder_name="${dir%/}"
    
    # List the files in the folder and select the first one
    # file_to_copy="$(ls -1 "$dir" | head -n 1)"

    file_name="${folder_name}/nontextured.stl"
    new_name=`echo $folder_name | sed 's/models\/ycb\///' | sed 's/.\{11\}$//'`
    new_name="ycb/${new_name}.stl"
    
    # echo $file_name
    # echo $new_name
    
    # Copy the selected file to the parent directory with the folder's name
    cp "${file_name}" "${new_name}"
done
