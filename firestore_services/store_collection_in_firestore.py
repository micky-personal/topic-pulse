from google.cloud import datastore

db = datastore.Client()

def store_collection_in_firestore(collection_name, firestore_document):
    try:
        # Create a new Datastore Entity and generate a new key.
        key = db.key(collection_name)
        entity = datastore.Entity(key=key)
        entity.update(firestore_document)

        # Use db.put() to store the entity in Datastore.
        db.put(entity)

        # Print the auto-generated ID after the put operation.
        print(f"Successfully stored data to Firestore in Datastore Mode in "
              f"collection '{collection_name}' with document ID '{entity.key.id}'.")

    except Exception as e:
        print(f"An error occurred: {e}")
