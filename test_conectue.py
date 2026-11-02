import json
from websocket import create_connection

ws = create_connection("ws://127.0.0.1:30020")

message = {
    "MessageName": "http",
    "Parameters": {
        "Url": "/remote/object/property",
        "Verb": "PUT",
        "Body": {
            "ObjectPath": "/Memory/UEDPIE_0_Untitled_MainGrid_L0_X0_Y-1.Untitled:PersistentLevel.Spawn_C_UAID_00155D02090057EF02_1296750191",
            "propertyName": "flag",
            "access": "WRITE_ACCESS",
            "propertyValue": {
                "flag": True
            }
        }
    }
}
ws.send(json.dumps(message))

print(ws.recv())


ws.close()