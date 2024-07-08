## server
The response from the server have to include {status code, message, data}

## client
The client need to consider the following question:

1 The connection to the server, set up the retry machenism if nessary.

2 Get the message successfully? 

4 Deal with the data based on the status code.

5 return {-1, error message] if you don't konw the error type.
