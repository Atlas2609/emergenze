% ============================================================
% Base di Conoscenza Prolog — Mappa Emergenze Urbane
% Sistema di Gestione Intelligente delle Emergenze
%
% Fatti (mappa, mezzi, compatibilità) e regole (clausole di
% Horn) da cui il motore Prolog deriva transitabilità e costi
% degli archi a partire dalle condizioni meteo correnti.
% ============================================================

% --- Dichiarazione dei predicati dinamici ---
% Questi fatti possono essere asseriti/ritrattati a runtime
% da Python tramite pyswip, simulando eventi in tempo reale.
:- dynamic condizione_meteo/2.
:- dynamic emergenza_attiva/4.
:- dynamic mezzo_occupato/1.
:- dynamic fascia_oraria_corrente/1.

% ============================================================
% FATTI BASE
% ============================================================

% --- Nodi della mappa cittadina (15 nodi) ---
nodo(ospedale).
nodo(stazione_pompieri).
nodo(piazza_centrale).
nodo(incrocio_nord).
nodo(incrocio_sud).
nodo(incrocio_est).
nodo(incrocio_ovest).
nodo(zona_industriale).
nodo(porto).
nodo(aeroporto).
nodo(universita).
nodo(stadio).
nodo(parco).
nodo(centro_commerciale).
nodo(stazione_treni).

% --- Coordinate dei nodi (X, Y) per euristica A* ---
coordinate(ospedale, 2, 8).
coordinate(stazione_pompieri, 8, 8).
coordinate(piazza_centrale, 5, 5).
coordinate(incrocio_nord, 5, 9).
coordinate(incrocio_sud, 5, 1).
coordinate(incrocio_est, 9, 5).
coordinate(incrocio_ovest, 1, 5).
coordinate(zona_industriale, 9, 2).
coordinate(porto, 10, 0).
coordinate(aeroporto, 0, 10).
coordinate(universita, 3, 3).
coordinate(stadio, 7, 3).
coordinate(parco, 3, 7).
coordinate(centro_commerciale, 7, 7).
coordinate(stazione_treni, 5, 0).

% --- Archi del grafo (Nodo1, Nodo2, Distanza) ---
% Grafo connesso con cicli multipli. Vincolo di progetto: la distanza
% di ogni arco e' >= della distanza euclidea tra le coordinate dei due
% nodi, cosi' l'euristica euclidea di A* e' ammissibile e consistente
% (verificato da test_kb.test_distanze_compatibili_con_euristica).
arco(ospedale, incrocio_nord, 4).
arco(ospedale, parco, 2).
arco(ospedale, incrocio_ovest, 4).
arco(incrocio_nord, piazza_centrale, 4).
arco(incrocio_nord, aeroporto, 6).
arco(incrocio_nord, centro_commerciale, 3).
arco(piazza_centrale, incrocio_est, 4).
arco(piazza_centrale, incrocio_ovest, 4).
arco(piazza_centrale, incrocio_sud, 4).
arco(piazza_centrale, universita, 3).
arco(piazza_centrale, parco, 3).
arco(incrocio_est, centro_commerciale, 3).
arco(incrocio_est, zona_industriale, 3).
arco(incrocio_est, stadio, 3).
arco(incrocio_ovest, parco, 3).
arco(incrocio_ovest, aeroporto, 6).
arco(incrocio_ovest, universita, 3).
arco(incrocio_sud, stazione_treni, 2).
arco(incrocio_sud, universita, 3).
arco(incrocio_sud, stadio, 3).
arco(zona_industriale, porto, 3).
arco(zona_industriale, stadio, 3).
arco(stazione_pompieri, centro_commerciale, 2).
arco(stazione_pompieri, incrocio_est, 4).
arco(stazione_treni, porto, 5).
arco(stadio, centro_commerciale, 5).

% --- Bidirezionalità degli archi ---
% Regola deduttiva: se esiste arco(X,Y,D) allora esiste anche arco(Y,X,D)
arco_bidirezionale(X, Y, D) :- arco(X, Y, D).
arco_bidirezionale(X, Y, D) :- arco(Y, X, D).

% --- Zone della città ---
% Ogni nodo appartiene a una zona geografica
zona(ospedale, zona_nord).
zona(incrocio_nord, zona_nord).
zona(aeroporto, zona_nord).
zona(parco, zona_nord).
zona(piazza_centrale, zona_centro).
zona(universita, zona_centro).
zona(incrocio_est, zona_centro).
zona(incrocio_ovest, zona_centro).
zona(incrocio_sud, zona_sud).
zona(stazione_treni, zona_sud).
zona(stadio, zona_sud).
zona(zona_industriale, zona_sud).
zona(porto, zona_sud).
zona(centro_commerciale, zona_est).
zona(stazione_pompieri, zona_est).

% --- Tipi di veicoli e classificazione ---
% tipo_veicolo(NomeMezzo, Categoria)
tipo_veicolo(ambulanza, leggero).
tipo_veicolo(autopompa, mezzo_pesante).
tipo_veicolo(autoscala, mezzo_pesante).

% --- Capacità dei mezzi (numero massimo di emergenze gestibili) ---
capacita_mezzo(ambulanza, 1).
capacita_mezzo(autopompa, 2).
capacita_mezzo(autoscala, 1).

% --- Compatibilità mezzo-emergenza ---
% Definisce quali tipi di mezzo possono rispondere a quali tipi di emergenza
compatibile(ambulanza, medica).
compatibile(ambulanza, incidente).
compatibile(autopompa, incendio).
compatibile(autopompa, allagamento).
compatibile(autopompa, incidente).
compatibile(autoscala, incendio).
compatibile(autoscala, soccorso_altezza).

% --- Tipi di emergenza gestiti dal sistema ---
tipo_emergenza(medica).
tipo_emergenza(incendio).
tipo_emergenza(allagamento).
tipo_emergenza(incidente).
tipo_emergenza(soccorso_altezza).

% --- Basi operative dei mezzi: base_mezzo(TipoMezzo, Nodo) ---
base_mezzo(ambulanza, ospedale).
base_mezzo(ambulanza, piazza_centrale).
base_mezzo(ambulanza, stazione_treni).
base_mezzo(autopompa, stazione_pompieri).
base_mezzo(autopompa, zona_industriale).
base_mezzo(autoscala, stazione_pompieri).

% --- Fascia oraria corrente (fatto dinamico, impostato dallo scenario) ---
fascia_oraria_corrente(mattina).

% --- Condizioni meteo iniziali (fatti dinamici) ---
% Possono essere modificate a runtime per simulare cambiamenti
condizione_meteo(zona_nord, sereno).
condizione_meteo(zona_centro, pioggia).
condizione_meteo(zona_sud, sereno).
condizione_meteo(zona_est, nebbia).

% ============================================================
% REGOLE DEDUTTIVE (Clausole di Horn)
% I fatti derivati non sono memorizzati: sono dedotti a ogni query.
% ============================================================

% --- Regola 1: Strada allagata ---
% Una strada (nodo) è allagata se la zona in cui si trova
% ha condizione meteo "pioggia_forte" oppure "alluvione".
% Questo fatto NON è enumerato: è DERIVATO per deduzione
% dalla combinazione di zona/2 e condizione_meteo/2.
strada_allagata(Nodo) :-
    zona(Nodo, Zona),
    condizione_meteo(Zona, pioggia_forte).

strada_allagata(Nodo) :-
    zona(Nodo, Zona),
    condizione_meteo(Zona, alluvione).

% --- Regola 2: Visibilità ridotta ---
% Derivata dalla condizione meteo: nebbia o pioggia forte
visibilita_ridotta(Nodo) :-
    zona(Nodo, Zona),
    condizione_meteo(Zona, nebbia).

visibilita_ridotta(Nodo) :-
    zona(Nodo, Zona),
    condizione_meteo(Zona, pioggia_forte).

% --- Regola 3: Transitabilità di un arco ---
% Clausola di Horn con eccezione esplicita:
%   - Se nessun nodo è allagato: transitabile per TUTTI i veicoli
%   - Se un nodo è allagato: transitabile SOLO per mezzi pesanti
%   - L'ambulanza (leggero) NON può transitare su strade allagate

% Caso A: arco senza allagamento -> qualunque veicolo può transitare
transitabile(X, Y, TipoVeicolo) :-
    arco_bidirezionale(X, Y, _),
    tipo_veicolo(TipoVeicolo, _),
    \+ strada_allagata(X),
    \+ strada_allagata(Y).

% Caso B: arco con almeno un nodo allagato, ma veicolo è mezzo pesante
% Eccezione esplicita: il mezzo pesante può forzare il passaggio
transitabile(X, Y, TipoVeicolo) :-
    arco_bidirezionale(X, Y, _),
    tipo_veicolo(TipoVeicolo, mezzo_pesante),
    (strada_allagata(X) ; strada_allagata(Y)).

% --- Regola 4: Costo dell'arco (con maggiorazioni dedotte) ---
% Il costo non è un dato statico ma è CALCOLATO per deduzione
% combinando: distanza base, condizioni meteo, stato allagamento.
costo_arco(X, Y, TipoVeicolo, Costo) :-
    arco_bidirezionale(X, Y, DistanzaBase),
    transitabile(X, Y, TipoVeicolo),
    calcola_maggiorazione(X, Y, TipoVeicolo, DistanzaBase, Costo).

% --- Calcolo composito della maggiorazione ---
calcola_maggiorazione(X, Y, TipoVeicolo, Base, Costo) :-
    fattore_meteo(X, FX),
    fattore_meteo(Y, FY),
    fattore_allagamento(X, Y, TipoVeicolo, FA),
    FMax is max(FX, FY),
    Costo is Base * FMax * FA.

% --- Fattore meteo per nodo ---
% Nebbia: +50% (visibilità ridotta), pioggia: +30%,
% pioggia forte: +40%, alluvione: +60%
% Altrimenti: nessun rallentamento
fattore_meteo(Nodo, 1.5) :-
    zona(Nodo, Zona),
    condizione_meteo(Zona, nebbia), !.
fattore_meteo(Nodo, 1.3) :-
    zona(Nodo, Zona),
    condizione_meteo(Zona, pioggia), !.
fattore_meteo(Nodo, 1.4) :-
    zona(Nodo, Zona),
    condizione_meteo(Zona, pioggia_forte), !.
fattore_meteo(Nodo, 1.6) :-
    zona(Nodo, Zona),
    condizione_meteo(Zona, alluvione), !.
fattore_meteo(_, 1.0).

% --- Fattore allagamento ---
% Mezzo pesante su strada allagata: costo raddoppiato (può passare, ma lentamente)
% Altrimenti: nessuna penalità aggiuntiva
fattore_allagamento(X, Y, TipoVeicolo, 2.0) :-
    tipo_veicolo(TipoVeicolo, mezzo_pesante),
    (strada_allagata(X) ; strada_allagata(Y)), !.
fattore_allagamento(_, _, _, 1.0).

% ============================================================
% PREDICATI DI UTILITA'
% ============================================================

% --- Distanza euclidea tra due nodi (per euristica A*) ---
distanza_euclidea(N1, N2, Dist) :-
    coordinate(N1, X1, Y1),
    coordinate(N2, X2, Y2),
    DX is X2 - X1,
    DY is Y2 - Y1,
    Dist is sqrt(DX * DX + DY * DY).

% --- Ottenere i vicini di un nodo ---
vicino(Nodo, Vicino, Distanza) :-
    arco_bidirezionale(Nodo, Vicino, Distanza).

% --- Verifica compatibilità mezzo-emergenza ---
mezzo_compatibile(TipoMezzo, TipoEmergenza) :-
    compatibile(TipoMezzo, TipoEmergenza).

% --- Meteo in un nodo: dedotto dalla zona di appartenenza ---
meteo_nodo(Nodo, Condizione) :-
    zona(Nodo, Zona),
    condizione_meteo(Zona, Condizione).

% --- Emergenza critica: gravita' alta in un nodo con meteo avverso ---
% Usata per segnalare all'operatore i casi in cui la KB, prima ancora
% del modello ML, suggerisce di inviare rinforzi.
meteo_avverso(pioggia_forte).
meteo_avverso(alluvione).
meteo_avverso(nebbia).

emergenza_critica(Nodo, alta) :-
    meteo_nodo(Nodo, Condizione),
    meteo_avverso(Condizione).
