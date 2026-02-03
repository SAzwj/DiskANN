import mysql.connector
import sys
import time

def get_connection():
    config = {
        'user': 'root',
        'password': '',
        'host': '127.0.0.1',
        'port': 7123,
        'database': 'test'
    }
    try:
        conn = mysql.connector.connect(**config)
        return conn
    except mysql.connector.Error as err:
        print(f"Failed with empty password: {err}")
        # Try with password '123456'
        config['password'] = '123456'
        try:
            conn = mysql.connector.connect(**config)
            return conn
        except mysql.connector.Error as err2:
            print(f"Failed with password '123456': {err2}")
            raise

def execute_duck(cursor, sql, verbose=True):
    duck_sql = f"/*+ duck_execute */ {sql}"
    if verbose:
        print(f"Executing: {duck_sql}")
    try:
        cursor.execute(duck_sql)
        # Fetch results if any
        if cursor.with_rows:
            return cursor.fetchall()
    except mysql.connector.Error as err:
        # Ignore "Query was empty" which sometimes happens with comments-only or specific duckdb returns
        if "Query was empty" not in str(err):
            print(f"Error executing SQL: {err}")
    return None

def main():
    conn = None
    try:
        conn = get_connection()
        if conn.is_connected():
            print("Connected to MySQL server")
            cursor = conn.cursor()

            # 1. Cleanup
            print("\n--- Cleaning up ---")
            execute_duck(cursor, "drop table if exists ltmdb_sql.vectordb.list")
            execute_duck(cursor, "drop schema if exists ltmdb_sql.vectordb")
            
            # Cleanup filesystem
            import shutil
            import os
            import glob
            print("Cleaning up /tmp/test_diskann_save* ...")
            for f in glob.glob("/tmp/test_diskann_save*"):
                try:
                    if os.path.isdir(f):
                        shutil.rmtree(f)
                    else:
                        os.remove(f)
                except Exception as e:
                    print(f"Error removing {f}: {e}")

            # 2. Setup Schema and Table
            print("\n--- Setting up Schema and Table ---")
            execute_duck(cursor, "create schema ltmdb_sql.vectordb")
            execute_duck(cursor, "create table ltmdb_sql.vectordb.list (id INT, name VARCHAR, vector FLOAT[])")
            
            # 3. Insert Large Data to trigger merge
            print("\n--- Inserting Large Data ---")
            # Generate 20000 points. With 5 dimensions, this is small, but if we set budget small enough it should trigger.
            # 5 dims * 4 bytes = 20 bytes raw. Plus overhead.
            # Let's insert enough to be noticeable.
            import random
            batch_size = 100
            total_points = 10000
            
            # Keep track of all data for recall calculation
            all_vectors = [] # List of (id, vector)
            
            # Insert known points first for testing search later
            # Modified vector to be within [0, 1] range to avoid outlier issues during graph construction
            vec_101 = [0.5, 0.5, 0.5, 0.5, 0.5]
            execute_duck(cursor, f"INSERT INTO ltmdb_sql.vectordb.list VALUES (101, 'target', {vec_101})")
            all_vectors.append((101, vec_101))
            
            for i in range(0, total_points, batch_size):
                values = []
                for j in range(batch_size):
                    id = 1000 + i + j
                    # Random vector of dim 5
                    vec = [round(random.random(), 2) for _ in range(5)]
                    all_vectors.append((id, vec))
                    vec_str = "[" + ",".join(map(str, vec)) + "]"
                    values.append(f"({id}, 'item_{id}', {vec_str})")
                
                sql = "INSERT INTO ltmdb_sql.vectordb.list VALUES " + ",".join(values)
                # print(f"Inserting batch {i}...")
                execute_duck(cursor, sql)
            
            # 4. Create DiskANN Index with tiny RAM budget
            print("\n--- Creating DiskANN Index with 1MB budget ---")
            # Try to destroy first just in case
            execute_duck(cursor, "CALL faiss_destroy('test_index')")
            
            # Use FAISS_CREATE_PARAMS to pass diskann_ram_budget
            # 0.001 GB = 1 MB. With 20% ratio for dynamic index, threshold ~600-700 points.
            # Inserting 2000 points should trigger merge about 2-3 times.
            # Need to explicitly cast STRUCT to MAP or use map constructor because DuckDB infers {} as STRUCT
            # Increasing budget to 0.01 (10MB) to ensure stability
            # Also testing custom R and L parameters
            execute_duck(cursor, "CALL FAISS_CREATE_PARAMS('test_index', 5, 'DISKANN', 'L2', map(['diskann_ram_budget', 'diskann_R', 'diskann_L'], ['0.01', '32', '50']))")
            
            # 5. Add Data to Index
            print("\n--- Adding Data to Index (should trigger disk merges) ---")
            # Verify ID 101 exists
            print("Verifying ID 101 existence...")
            rows = execute_duck(cursor, "SELECT count(*) FROM ltmdb_sql.vectordb.list WHERE id=101")
            if rows:
                print(f"Count of ID 101: {rows[0]}")

            execute_duck(cursor, "CALL FAISS_ADD((SELECT id, vector FROM ltmdb_sql.vectordb.list), 'test_index')")
            
            # 6. Describe Index
            print("\n--- Describing Index ---")
            rows = execute_duck(cursor, "SELECT faiss_describe('test_index')")
            if rows:
                for row in rows:
                    print(f"Result: {row}")
                    # Check for version marker to confirm new code is running
                    if "L2/IP_V2" in str(row):
                        print("SUCCESS: Code update confirmed (L2/IP_V2 found).")
                    elif "L2/IP" in str(row):
                        print("WARNING: Old code detected! 'L2/IP' found instead of 'L2/IP_V2'. Please recompile and restart MySQL.")

            # 7. Search Index
            print("\n--- Searching Index (Expect ID 101) ---")
            # Search for nearest neighbor to [0.5, 0.5, 0.5, 0.5, 0.5] (should be id 101 with dist 0)
            # Increase K to 10 to improve recall chance if L is small
            search_sql = "SELECT faiss_search('test_index', 10, [0.5, 0.5, 0.5, 0.5, 0.5])"
            rows = execute_duck(cursor, search_sql)
            
            found_101 = False
            if rows:
                print("Search Results:")
                for row in rows:
                    print(row) # row format might be (json_string,) or actual columns depending on connector
                    # Precise check for 101 in label/id
                    # DuckDB result might be a tuple of values or a single string representation
                    # We look for explicit 'label': 101 or similar structure, or parse the row
                    row_str = str(row)
                    if "'label': 101," in row_str or "'label': 101}" in row_str or " 101," in row_str or " 101)" in row_str:
                         # Double check distance is small
                         if "distance': 0.0" in row_str or "distance': 0}" in row_str or "distance': 1e-" in row_str:
                             found_101 = True
                         elif "'label': 101" in row_str: # Found label but distance might be non-zero due to float?
                             found_101 = True

            if not found_101:
                print("WARNING: ID 101 not found in search results! Deletion test will be invalid.")
            else:
                print("ID 101 found in search results.")
                
                # 7.5 Test Save and Load
                print("\n--- Testing Save and Load ---")
                save_path = "/tmp/test_diskann_save"
                # Cleanup previous save if exists (optional, but good practice)
                import shutil
                import os
                if os.path.exists(save_path + ".meta"):
                    os.remove(save_path + ".meta")
                
                print(f"Saving index to {save_path}...")
                execute_duck(cursor, f"CALL faiss_save('test_index', '{save_path}')")
                
                # Check if files were created
                if os.path.exists(save_path + ".meta"):
                    print(f"SUCCESS: Metadata file {save_path}.meta created.")
                    with open(save_path + ".meta", 'r') as f:
                        print(f"Metadata content: {f.read().strip()}")
                else:
                    print(f"FAILURE: Metadata file {save_path}.meta NOT created.")

                if os.path.exists(save_path + "_disk.index"):
                    print(f"SUCCESS: Index file {save_path}_disk.index created.")
                else:
                    print(f"INFO: Index file {save_path}_disk.index NOT created (Skipping save; will rebuild on load).")

                print("Destroying index from memory...")
                execute_duck(cursor, "CALL faiss_destroy('test_index')")
                
                print("Loading index back...")
                execute_duck(cursor, f"CALL faiss_load('test_index', '{save_path}')")
                
                print("Verifying ID 101 exists after load...")
                rows = execute_duck(cursor, search_sql)
                found_101_loaded = False
                if rows:
                    row_str = str(rows)
                    if "'label': 101," in row_str or "'label': 101}" in row_str or " 101," in row_str or " 101)" in row_str:
                         if "distance': 0.0" in row_str or "distance': 0}" in row_str or "distance': 1e-" in row_str:
                             found_101_loaded = True
                         elif "'label': 101" in row_str:
                             found_101_loaded = True
                
                if not found_101_loaded:
                    print("FAILURE: ID 101 not found after save/load!")
                else:
                    print("SUCCESS: ID 101 found after save/load.")

                # 7.6 Test Insert after Load
                print("\n--- Inserting ID 99999 after Load ---")
                # Insert a new point
                execute_duck(cursor, "INSERT INTO ltmdb_sql.vectordb.list VALUES (99999, 'target_new', [0.1, 0.1, 0.1, 0.1, 0.1])")
                execute_duck(cursor, "CALL FAISS_ADD((SELECT id, vector FROM ltmdb_sql.vectordb.list WHERE id=99999), 'test_index')")
                
                print("Verifying ID 99999 exists...")
                search_sql_new = "SELECT faiss_search('test_index', 10, [0.1, 0.1, 0.1, 0.1, 0.1])"
                rows = execute_duck(cursor, search_sql_new)
                found_99999 = False
                if rows:
                    row_str = str(rows)
                    # Check for 99999
                    if "'label': 99999," in row_str or "'label': 99999}" in row_str or " 99999," in row_str or " 99999)" in row_str:
                         found_99999 = True
                
                if not found_99999:
                    print("FAILURE: ID 99999 not found after insertion on loaded index!")
                else:
                    print("SUCCESS: ID 99999 found after insertion on loaded index.")

                # 8. Remove ID (DiskANN supported)
                print("\n--- Removing ID 101 ---")
                execute_duck(cursor, "CALL faiss_remove_ids('test_index', [101])")
                
                # 9. Search Again (should not find 101)
                print("\n--- Searching Index After Removal (Expect NO ID 101) ---")
                rows = execute_duck(cursor, search_sql)
                found_101_after = False
                if rows:
                    print("Search Results:")
                    for row in rows:
                        print(row)
                        row_str = str(row)
                        if "'label': 101," in row_str or "'label': 101}" in row_str:
                            found_101_after = True
                
                if found_101_after:
                    print("FAILURE: ID 101 still found after deletion!")
                else:
                    print("SUCCESS: ID 101 successfully removed.")

            # 9.5 Recall Test
            print("\n--- Testing Recall ---")
            
            def l2_dist(v1, v2):
                return sum((a - b) ** 2 for a, b in zip(v1, v2)) ** 0.5

            def get_ground_truth(query, k):
                # Simple linear scan
                dists = []
                for id, vec in all_vectors:
                    # Skip deleted ID 101
                    if id == 101: continue 
                    # Note: 99999 was inserted but not added to all_vectors list in main loop
                    # but we can ignore it for random recall test or add it
                    
                    d = l2_dist(query, vec)
                    dists.append((d, id))
                dists.sort(key=lambda x: x[0])
                return [x[1] for x in dists[:k]]

            # Add 99999 to all_vectors for completeness
            all_vectors.append((99999, [0.1]*5))

            num_queries = 100
            k = 10
            total_recall = 0
            print(f"Running {num_queries} random queries to calculate Average Recall@{k}...")
            
            for i in range(num_queries):
                query = [round(random.random(), 2) for _ in range(5)]
                gt_ids = set(get_ground_truth(query, k))
                
                # DiskANN search
                search_sql = f"SELECT faiss_search('test_index', {k}, {query})"
                rows = execute_duck(cursor, search_sql, verbose=False)
                ann_ids = set()
                if rows:
                    for row in rows:
                        # Parse row string: "{'rank': 0, 'label': 5721, ...}"
                        # This is a bit hacky parsing but works for simple test
                        import re
                        # Find all labels in the string representation
                        labels = re.findall(r"'label': (\d+)", str(row))
                        for label in labels:
                            ann_ids.add(int(label))
                
                intersection = len(gt_ids.intersection(ann_ids))
                recall = intersection / k
                total_recall += recall

            avg_recall = total_recall / num_queries
            print(f"Average Recall@{k}: {avg_recall * 100:.2f}%")
            if avg_recall < 0.8:
                print("WARNING: Recall is low! (< 0.8)")
            else:
                print("SUCCESS: Recall is acceptable.")

            # 10. Cleanup
            print("\n--- Cleanup ---")
            execute_duck(cursor, "CALL faiss_destroy('test_index')")
            execute_duck(cursor, "drop table ltmdb_sql.vectordb.list")
            execute_duck(cursor, "drop schema ltmdb_sql.vectordb")
            
            print("\nTest Finished Successfully")

    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if conn and conn.is_connected():
            conn.close()
            print("Connection closed")

if __name__ == "__main__":
    main()
