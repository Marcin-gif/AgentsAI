def insert_invoice_to_db(json_data, db_path="faktury.db"):
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    def insert_podmiot(table, dane):
        cursor.execute(f"""
            INSERT INTO {table} (nazwa, ulica, kod_pocztowy, miasto, nip, email)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            dane.get("nazwa"),
            dane.get("ulica"),
            dane.get("kod_pocztowy"),
            dane.get("miasto"),
            dane.get("nip"),
            dane.get("email") if table == "sprzedawcy" else None
        ))
        return cursor.lastrowid

    def insert_faktura(dane, s_id, k_id, o_id):
        cursor.execute("""
            INSERT INTO faktury (
                numer_faktury, data_wystawienia, data_sprzedazy,
                sposob_zaplata, termin_zaplata, po_number,
                sprzedawca_id, kupujacy_id, odbiorca_id,
                zaplacono, pozostalo_do_zaplaty,
                kwota_do_zaplaty_koncowa, slownie, uwagi
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            dane["numer_faktury"],
            dane["data_wystawienia"],
            dane["data_sprzedazy"],
            dane["sposob_zaplata"],
            dane["termin_zaplata"],
            dane["po_number"],
            s_id, k_id, o_id,
            dane.get("zaplacono"),
            dane.get("pozostalo_do_zaplaty"),
            dane.get("kwota_do_zaplaty_koncowa"),
            dane.get("slownie"),
            dane.get("uwagi")
        ))
        return cursor.lastrowid

    def insert_pozycje(faktura_id, pozycje):
        for poz in pozycje:
            cursor.execute("""
                INSERT INTO pozycje_faktury (
                    faktura_id, lp, nazwa_towaru_uslugi, rabat_procent, ilosc,
                    jednostka_miary, cena_netto_przed_rabatem, cena_netto_po_rabacie,
                    wartosc_netto_pozycji, stawka_vat_procent, kwota_vat_pozycji, wartosc_brutto_pozycji
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                faktura_id,
                int(poz["lp"]),
                poz["nazwa_towaru_uslugi"],
                poz["rabat_procent"],
                float(poz["ilosc"]),
                poz["jednostka_miary"],
                float(poz["cena_netto_przed_rabatem"]),
                float(poz["cena_netto_po_rabacie"]),
                float(poz["wartosc_netto_pozycji"]),
                float(poz["stawka_vat_procent"]),
                float(poz["kwota_vat_pozycji"]),
                float(poz["wartosc_brutto_pozycji"])
            ))

    def insert_podsumowanie(faktura_id, podsumowanie):
        cursor.execute("""
            INSERT INTO podsumowania (
                faktura_id, wartosc_netto_sumaryczna,
                stawka_vat_sumaryczna_procent,
                wartosc_vat_sumaryczna,
                wartosc_brutto_sumaryczna,
                waluta
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            faktura_id,
            podsumowanie["wartosc_netto_sumaryczna"],
            podsumowanie["stawka_vat_sumaryczna_procent"],
            podsumowanie["wartosc_vat_sumaryczna"],
            podsumowanie["wartosc_brutto_sumaryczna"],
            podsumowanie["waluta"]
        ))

    # Wstaw wszystko
    s_id = insert_podmiot("sprzedawcy", json_data["sprzedawca"])
    k_id = insert_podmiot("kupujacy", json_data["kupujacy"])
    o_id = insert_podmiot("odbiorcy", json_data["odbiorca"])
    f_id = insert_faktura(json_data["faktura"], s_id, k_id, o_id)
    insert_pozycje(f_id, json_data["pozycje"])
    insert_podsumowanie(f_id, json_data["podsumowanie"])

    conn.commit()
    conn.close()
