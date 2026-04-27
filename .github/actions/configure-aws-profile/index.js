const { execSync } = require('child_process');

try {
    const profile = process.env['INPUT_PROFILE'];
    const region = process.env['INPUT_AWS-REGION'];
    const accessKeyId = process.env['INPUT_AWS-ACCESS-KEY-ID'];
    const secretAccessKey = process.env['INPUT_AWS-SECRET-ACCESS-KEY'];
    const sessionToken = process.env['INPUT_AWS-SESSION-TOKEN'];

    const missingInputs = [];
    if (!profile) {
        missingInputs.push('profile');
    }
    if (!region) {
        missingInputs.push('region');
    }
    if (!accessKeyId) {
        missingInputs.push('accessKeyId');
    }
    if (!secretAccessKey) {
        missingInputs.push('secretAccessKey');
    }
    if (!sessionToken) {
        missingInputs.push('sessionToken');
    }
    if (missingInputs.length > 0) {
        throw new Error(`Missing required input(s): ${missingInputs}`);
    }

    console.log(`Configuring AWS Profile ${profile}`);

    const commands = [
        `aws configure set --profile ${profile} region ${region}`,
        `aws configure set --profile ${profile} aws_access_key_id ${accessKeyId}`,
        `aws configure set --profile ${profile} aws_secret_access_key ${secretAccessKey}`,
        `aws configure set --profile ${profile} aws_session_token ${sessionToken}`,
    ];

    // Execute each command synchronously
    for (const command of commands) {
        execSync(command);
    }

    console.log('AWS profile configuration completed successfully');
} catch (error) {
    console.error(error.message);
    process.exit(1);
}
