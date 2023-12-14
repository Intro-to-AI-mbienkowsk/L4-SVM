# WSI 2023 LAB3 - Gry o sumie zerowej
### Maksym Bieńkowski

# Zawartość archiwum


## /src/

----
- `util.py` - zawiera funkcję importującą i przetwarzającą dane oraz stałe wykorzystywane w obliczeniach
- `PrimalSVM.py` - implementacja maszyny wektorów nośnych w formie pierwotnej
- `DualSVM.py` - implementacja maszyny wektorów nośnych w formie dualnej
- `MnistClassifier.py` - klasa agregująca wyniki 10 maszyn wektorów nośnych, odpowiedzialna za rozpoznawanie cyfr


## uruchamialne skrypty

----
Ze względu na charakterystykę zadania (dość długie czasy trenowania maszyn), zdecydowałem się nie pisać w tym tygodniu 
interaktywnego skryptu, ponieważ aby osiągnąć sensowne wyniki, programy potrzebują przynajmniej kilku minut. Możliwe jest
uruchomienie `main.py`, który zawiera konfigurację osiągającą około 85%, jednakże potrzebującą około 5 minut na przeprowadzenie obliczeń.

## Krótki opis rozwiązania

---
Zaimplementowane są dwa warianty maszyny wektorów nośnych, forma dualna oraz pierwotna. Klasyfikator agreguje funkcje decyzji 10 maszyn i na tej podstawie przypisuje obrazowi etykietę. Szczegóły implementacyjne opisane są w sprawozdaniu.