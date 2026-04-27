const fs = require('fs');
const path = require('path');
const os = require('os');

function deleteFolderRecursive(folderPath) {
    try {
        if (fs.existsSync(folderPath)) {
            fs.readdirSync(folderPath).forEach((file) => {
                const curPath = path.join(folderPath, file);
                try {
                    if (fs.lstatSync(curPath).isDirectory()) {
                        deleteFolderRecursive(curPath);
                    } else {
                        fs.unlinkSync(curPath);
                    }
                } catch (err) {
                    console.warn(`Warning: Failed to delete ${curPath}: ${err.message}`);
                }
            });
            fs.rmdirSync(folderPath);
        }
    } catch (err) {
        console.warn(`Warning: Error processing ${folderPath}: ${err.message}`);
    }
}

try {
    const homeDir = os.homedir();
    const awsFolder = path.join(homeDir, '.aws');

    console.log(`Cleaning up AWS credentials folder: ${awsFolder}`);

    deleteFolderRecursive(awsFolder);

    console.log('AWS credentials folder cleanup completed successfully');
} catch (error) {
    console.error(`Error during cleanup: ${error.message}`);
    // TODO don't exit with error for cleanup issues
    process.exit(1);
}
