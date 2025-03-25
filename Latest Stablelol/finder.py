# /home/wayne/PycharmProjects/Theory/.venv/list_backends.py
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(channel="ibm_quantum")
backends = service.backends(min_num_qubits=127, operational=True, simulator=False)
print("Available 127-qubit backends:", [b.name for b in backends])